import torch.nn.functional as F
import torch
import math
import numpy as np
from asgdiff.models.noise_schedule import *
from tqdm import tqdm
from itertools import chain

"""
"The continuous-time masked diffusion core is adapted for mixed-type tabular data with categorical dimensions of different cardinalities."
"""

S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1

class UnifiedCtimeDiffusion(torch.nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            denoise_fn,
            y_only_model,
            num_timesteps=1000,
            scheduler='power_mean',
            cat_scheduler='log_linear',
            noise_dist='uniform',
            edm_params={},
            noise_dist_params={},
            noise_schedule_params={},
            sampler_params={},
            device=torch.device('cpu'),
            **kwargs
        ):

        super(UnifiedCtimeDiffusion, self).__init__()

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device) if len(num_classes)>0 else torch.tensor([]).to(device).int()
        self.mask_index = torch.tensor(self.num_classes).long().to(device)
        self.neg_infinity = -1000000.0 
        self.num_classes_w_mask = tuple(self.num_classes + 1)

        offsets = np.cumsum(self.num_classes)
        offsets = np.append([0], offsets)
        self.slices_for_classes = []
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(offsets).to(device)
        
        offsets = np.cumsum(self.num_classes) + np.arange(1, len(self.num_classes)+1)
        offsets = np.append([0], offsets)
        self.slices_for_classes_with_mask = []
        for i in range(1, len(offsets)):
            self.slices_for_classes_with_mask.append(np.arange(offsets[i - 1], offsets[i]))

        self._denoise_fn = denoise_fn
        self.y_only_model = y_only_model
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.cat_scheduler = cat_scheduler
        self.noise_dist = noise_dist
        self.edm_params = edm_params
        self.noise_dist_params = noise_dist_params
        self.sampler_params = sampler_params
        if self.num_numerical_features == 0:
            self.sampler_params['stochastic_sampler'] = False
            self.sampler_params['second_order_correction'] = False
        
        self.w_num = 0.0
        self.w_cat = 0.0
        self.num_mask_idx = []
        self.cat_mask_idx = []
        
        self.device = device
        
        if self.scheduler == 'power_mean':
            self.num_schedule = PowerMeanNoise(**noise_schedule_params)
        elif self.scheduler == 'power_mean_per_column':
            self.num_schedule = PowerMeanNoise_PerColumn(num_numerical = num_numerical_features, **noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.scheduler}-- is not implemented for contiuous data at CTIME ")
        
        if self.cat_scheduler == 'log_linear':
            self.cat_schedule = LogLinearNoise(**noise_schedule_params)
        elif self.cat_scheduler == 'log_linear_per_column':
            self.cat_schedule = LogLinearNoise_PerColumn(num_categories = len(num_classes), **noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.cat_scheduler}-- is not implemented for discrete data at CTIME ")
        
        self.debug_flag_once = None

    def set_fairness_config(self, sens_num_idx, sens_cat_idx, target_num_idx=[], target_cat_idx=[], lambda_adv=1.0, alpha_consistency=1.0):
        self.sens_num_idx = sens_num_idx
        self.sens_cat_idx = sens_cat_idx
        self.target_num_idx = target_num_idx
        self.target_cat_idx = target_cat_idx
        self.lambda_adv = lambda_adv
        self.alpha_consistency = alpha_consistency
        self.fairness_enabled = True


    def mixed_loss(self, x):
        b = x.shape[0]
        device = x.device

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:].long()
        
        # Check fairness flag
        fairness = getattr(self, 'fairness_enabled', False)
        
        # Sample noise level
        if self.noise_dist == "uniform_t":
            t = torch.rand(b, device=device, dtype=x_num.dtype)
            t = t[:, None]
            sigma_num = self.num_schedule.total_noise(t)
            sigma_cat = self.cat_schedule.total_noise(t)
            dsigma_cat = self.cat_schedule.rate_noise(t)
        else:
            sigma_num = self.sample_ctime_noise(x)       
            t = self.num_schedule.inverse_to_t(sigma_num)
            while torch.any((t < 0) + (t > 1)):     
                # restrict t to [0,1]
                # this iterative approach is equivalent to sampling from a truncated version of the orignal noise distribution
                invalid_idx = ((t < 0) + (t > 1)).nonzero().squeeze(-1)
                sigma_num[invalid_idx] = self.sample_ctime_noise(x[:len(invalid_idx)])
                t = self.num_schedule.inverse_to_t(sigma_num)
            assert not torch.any((t < 0) + (t > 1))
            sigma_cat = self.cat_schedule.total_noise(t)
            
        # Convert sigma_cat to the corresponding alpha and move_chance
        alpha = torch.exp(-sigma_cat)
        move_chance = -torch.expm1(-sigma_cat)      # torch.expm1 gives better numertical stability
        
        # Prepare Batch for SCD (Siamese Counterfactual Denoising)
        if fairness:
            # Construct Counterfactual Batch
            x_num_cf = x_num.clone()
            x_cat_cf = x_cat.clone()
            
            # Flip sensitive attributes
            for idx in self.sens_cat_idx:
                K = self.num_classes[idx]
                if K <= 1: continue 
                
                if K == 2:
                    x_cat_cf[:, idx] = 1 - x_cat_cf[:, idx]
                else:
                    x_cat_cf[:, idx] = (x_cat_cf[:, idx] + 1) % K
            
            # Concatenate real and cf: [Real; CF]
            x_num_in = torch.cat([x_num, x_num_cf], dim=0)
            x_cat_in = torch.cat([x_cat, x_cat_cf], dim=0)
            
            # Replicate Noise parameters
            sigma_num_in = torch.cat([sigma_num, sigma_num], dim=0)
            t_in = torch.cat([t, t], dim=0)
        else:
            x_num_in = x_num
            x_cat_in = x_cat
            sigma_num_in = sigma_num
            t_in = t
            
        # Continuous forward diff
        x_num_t = x_num_in
        if x_num_in.shape[1] > 0:
            noise = torch.randn_like(x_num) 
            if fairness:
                noise = torch.cat([noise, noise], dim=0) # Shared noise
            else:
                 noise = torch.randn_like(x_num_in)

            x_num_t = x_num_in + noise * sigma_num_in
        
        # Discrete forward diff
        x_cat_t = x_cat_in
        x_cat_t_soft = x_cat_in 
        
        if x_cat_in.shape[1] > 0:
            is_learnable = self.cat_scheduler == 'log_linear_per_column'
            strategy = 'soft'if is_learnable else 'hard'
            
            if fairness and strategy == 'hard':
                 move_indices = torch.rand(*x_cat.shape, device=device) < move_chance
                 move_indices = torch.cat([move_indices, move_indices], dim=0)
                 
                 xt = torch.where(move_indices, self.mask_index, x_cat_in)
                 xt_soft = self.to_one_hot(xt).to(move_chance.dtype)
                 x_cat_t, x_cat_t_soft = xt, xt_soft
            else:
                 if fairness: 
                     mc = torch.cat([move_chance, move_chance], dim=0)
                     x_cat_t, x_cat_t_soft = self.q_xt(x_cat_in, mc, strategy=strategy)
                 else:
                     x_cat_t, x_cat_t_soft = self.q_xt(x_cat_in, move_chance, strategy=strategy)

        # Predict original data
        out = self._denoise_fn(   
            x_num_t, x_cat_t_soft,
            t_in.squeeze(), sigma=sigma_num_in
        )
        if len(out) == 3:
            model_out_num, model_out_cat, pred_sens = out
        else:
            model_out_num, model_out_cat = out
            pred_sens = None

        d_loss = torch.zeros((1,), device=device).float()
        c_loss = torch.zeros((1,), device=device).float()
        adv_loss = torch.zeros((1,), device=device).float()
        consis_loss = torch.zeros((1,), device=device).float()

        # Split outputs if fairness
        if fairness:
            # First half is real, Second half is CF
            out_num_real, out_num_cf = torch.chunk(model_out_num, 2, dim=0)
            out_cat_real, out_cat_cf = torch.chunk(model_out_cat, 2, dim=0)
            
            task_out_num = out_num_real
            task_out_cat = out_cat_real
            task_x_num = x_num
            task_x_cat = x_cat
            task_sigma = sigma_num
            task_sigma_cat = sigma_cat
            task_dsigma_cat = dsigma_cat
            task_x_cat_t = x_cat_t[:b]
        else:
            task_out_num = model_out_num
            task_out_cat = model_out_cat
            task_x_num = x_num
            task_x_cat = x_cat
            task_sigma = sigma_num
            task_sigma_cat = sigma_cat
            task_dsigma_cat = dsigma_cat
            task_x_cat_t = x_cat_t

        if x_num.shape[1] > 0:
            c_loss = self._edm_loss(task_out_num, task_x_num, task_sigma)
        if x_cat.shape[1] > 0:
            logits = self._subs_parameterization(task_out_cat, task_x_cat_t)
            d_loss = self._absorbed_closs(logits, task_x_cat, task_sigma_cat, task_dsigma_cat)
            
        # Auxiliary Fairness Losses
        if fairness:
             # A. Adversarial Loss (AIB)
             if pred_sens is not None:
                 pred_sens_real, _ = torch.chunk(pred_sens, 2, dim=0)
                 
                 adv_loss_val = 0.0
                 current_out_idx = 0
                 
                 # Numerical targets
                 if len(self.sens_num_idx) > 0:
                     targets_num = x_num[:, self.sens_num_idx]
                     preds_num = pred_sens_real[:, :len(self.sens_num_idx)]
                     adv_loss_val += F.mse_loss(preds_num, targets_num)
                     current_out_idx += len(self.sens_num_idx)
                 
                 # Categorical targets
                 for i, cat_idx in enumerate(self.sens_cat_idx):
                     K_dim = self.num_classes[cat_idx]
                     
                     target_cat = x_cat[:, cat_idx]
                     pred_cat = pred_sens_real[:, current_out_idx : current_out_idx + K_dim]
                     adv_loss_val += F.cross_entropy(pred_cat, target_cat)
                     current_out_idx += K_dim
                     
                 adv_loss += self.lambda_adv * adv_loss_val
                 
             # B. Consistency Loss (SCD)
             # Numerical
             non_sens_num_mask = torch.ones(self.num_numerical_features, dtype=torch.bool, device=device)
             for idx in self.sens_num_idx:
                 non_sens_num_mask[idx] = False
             for idx in self.target_num_idx:
                 non_sens_num_mask[idx] = False
                 
             if non_sens_num_mask.any() and x_num.shape[1] > 0:
                 diff_num = (out_num_real[:, non_sens_num_mask] - out_num_cf[:, non_sens_num_mask])
                 loss_cons_num = (diff_num ** 2).mean()
                 consis_loss += self.alpha_consistency * loss_cons_num
             
             # Categorical
             if x_cat.shape[1] > 0:
                 mask_vec = torch.ones(model_out_cat.shape[1], device=device)
                 current = 0
                 for i, k_plus_1 in enumerate(self.num_classes_w_mask): 
                       if i in self.sens_cat_idx:
                            mask_vec[current : current + k_plus_1] = 0
                       if i in self.target_cat_idx:
                            mask_vec[current : current + k_plus_1] = 0
                       current += k_plus_1
                 
                 diff_cat = (out_cat_real - out_cat_cf) * mask_vec
                 loss_cons_cat = (diff_cat ** 2).mean()
                 consis_loss += self.alpha_consistency * loss_cons_cat

        return d_loss.mean(), c_loss.mean(), adv_loss.mean(), consis_loss.mean()

    @torch.no_grad()
    def sample(self, num_samples, guidance_scale=0.0):
        b = num_samples
        device = self.device
        dtype = torch.float32
        
        # Create the chain of t
        t = torch.linspace(0,1,self.num_timesteps, dtype=dtype, device=device)      # times = 0.0,...,1.0
        t = t[:, None]
        
        # Compute the chains of sigma
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]
        
        # Prepare sigma_hat for stochastic sampling mode
        if self.sampler_params['stochastic_sampler']:
            gamma = min(S_churn / self.num_timesteps, np.sqrt(2) - 1) * (S_min <= sigma_num_cur) * (sigma_num_cur <= S_max)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(t_hat, dim=-1, keepdim=True).values    # take the samllest t_hat induced by sigma_num
            zero_gamma = (gamma==0).any()
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat = t
            sigma_num_hat = sigma_num_cur
            sigma_cat_hat = sigma_cat_cur
                
        # Sample priors for the continuous dimensions
        z_norm = torch.randn((b, self.num_numerical_features), device=device) * sigma_num_cur[-1] 
            
        # Sample priors for the discrete dimensions
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros((b, 0), device=device).float()      # the default values for categorical sample if the dataset has no categorical entry
        if has_cat:
            z_cat = self._sample_masked_prior(
                b,
                len(self.num_classes),
            )
        
        pbar = tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps)
        pbar.set_description(f"Sampling Progress")
        for i in pbar:                  
            z_norm, z_cat, q_xs = self.edm_update(
                z_norm, z_cat, i, 
                t[i], t[i-1] if i > 0 else None, t_hat[i],
                sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i], 
                sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
                guidance_scale=guidance_scale
            )
        
        assert torch.all(z_cat < self.mask_index)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample
    
    def sample_all(self, num_samples, batch_size, keep_nan_samples=False, guidance_scale=0.0):        
        b = batch_size

        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            print(f"Samples left to generate: {num_samples-num_generated}")
            sample = self.sample(b, guidance_scale=guidance_scale)
            mask_nan = torch.any(sample.isnan(), dim=1)
            if keep_nan_samples:
                # If the sample instances that contains Nan are decided to be kept, the row with Nan will be foreced to all zeros
                sample = sample * (~mask_nan)[:, None]
            else:
                # Otherwise the instances with Nan will be eliminated
                sample = sample[~mask_nan]

            all_samples.append(sample)
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]

        return x_gen
    
    def q_xt(self, x, move_chance, strategy='hard'):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input. 
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        if strategy == 'hard':
            move_indices = torch.rand(
            * x.shape, device=x.device) < move_chance
            xt = torch.where(move_indices, self.mask_index, x)
            xt_soft = self.to_one_hot(xt).to(move_chance.dtype)
            return xt, xt_soft
        elif strategy == 'soft':
            bs = x.shape[0]
            xt_soft = torch.zeros(bs, torch.sum(self.mask_index+1), device=x.device)
            xt = torch.zeros_like(x)
            for i in range(len(self.num_classes)):
                slice_i = self.slices_for_classes_with_mask[i]
                # set the bernoulli probabilities, which determines the "coin flip" transition to the mask class
                prob_i = torch.zeros(bs, 2, device=x.device)
                prob_i[:,0] = 1-move_chance[:,i]
                prob_i[:,-1] = move_chance[:,i]
                log_prob_i = torch.log(prob_i)
                # draw soft samples and place them back to the corresponding columns
                soft_sample_i = F.gumbel_softmax(log_prob_i, tau=0.01, hard=True)
                idx = torch.stack((x[:,i]+slice_i[0], torch.ones_like(x[:,i])*slice_i[-1]), dim=-1)
                xt_soft[torch.arange(len(idx)).unsqueeze(1), idx] = soft_sample_i
                # retrieve the hard samples
                xt[:, i] = torch.where(soft_sample_i[:,1] > soft_sample_i[:,0], self.mask_index[i], x[:,i])
            return xt, xt_soft
    
    
    def _subs_parameterization(self, unormalized_prob, xt):
        # log prob at the mask index = - infinity
        unormalized_prob = self.pad(unormalized_prob, self.neg_infinity)
        
        unormalized_prob[:, range(unormalized_prob.shape[1]), self.mask_index] += self.neg_infinity
        
        # Take log softmax on the unnormalized probabilities to the logits
        logits = unormalized_prob - torch.logsumexp(unormalized_prob, dim=-1,
                                        keepdim=True)
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)    # (bs, K)
        logits[unmasked_indices] = self.neg_infinity 
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits
    
    def pad(self, x, pad_value):
        """
        Converts a concatenated tensor of class probabilities into a padded matrix, 
        where each sub-tensor is padded along the last dimension to match the largest 
        category size (max number of classes).

        Args:
            x (Tensor): The input tensor containing concatenated probabilities for all the categories in x_cat. 
                        [bs, sum(num_classes_w_mask)]
            pad_value (float): The value filled into the dummy entries, which are padded to ensure all sub-tensors have equal size 
                            along the last dimension.

        Returns:
            Tensor: A new tensorwith
                    [bs, len(num_classes_w_mask), max(num_classes_w_mask)), num_categories]
        """
        splited = torch.split(x, self.num_classes_w_mask, dim=-1)
        max_K = max(self.num_classes_w_mask)
        padded_ = [
            torch.cat((
                t, 
                pad_value*torch.ones(*(t.shape[:-1]), max_K-t.shape[-1], dtype=t.dtype, device=t.device)
            ), dim=-1) 
        for t in splited]
        out = torch.stack(padded_, dim=-2)
        return out
    
    def to_one_hot(self, x_cat):
        x_cat_oh = torch.cat(
            [F.one_hot(x_cat[:, i], num_classes=self.num_classes[i]+1,) for i in range(len(self.num_classes))], 
            dim=-1
        )
        return x_cat_oh
    
    def _absorbed_closs(self, model_output, x0, sigma, dsigma):
        """
            alpha: (bs,)
        """
        log_p_theta = torch.gather(
            model_output, -1, x0[:, :, None]
        ).squeeze(-1)
        alpha = torch.exp(-sigma)
        if self.cat_scheduler in ['log_linear_unified', 'log_linear_per_column']:
            elbo_weight = - dsigma / torch.expm1(sigma)
        else:
            elbo_weight = -1/(1-alpha)
        
        loss = elbo_weight * log_p_theta
        return loss
    
    def _sample_masked_prior(self, *batch_dims):
        return self.mask_index[None,:] * torch.ones(    
        * batch_dims, dtype=torch.int64, device=self.mask_index.device)
        
    def _masked_diffusion_update(self, log_p_x0, x, alpha_t, alpha_s):
        """
            # t: (bs,)
            log_p_x0: (bs, K, K_max)
            # alpha_t: (bs,)
            # alpha_s: (bs,)
            alpha_t: (bs, 1/K_cat)
            alpha_s: (bs,1/K_cat)
        """
        move_chance_t = 1 - alpha_t
        move_chance_s = 1 - alpha_s     
        move_chance_t = move_chance_t.unsqueeze(-1)
        move_chance_s = move_chance_s.unsqueeze(-1)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        # There is a noremalizing term is (1-\alpha_t) who's responsility is to ensure q_xs is normalized. 
        # However, omiting it won't make a difference for the Gumbel-max sampling trick in  _sample_categorical()
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        q_xs[:, range(q_xs.shape[1]), self.mask_index] = move_chance_s[:, :, 0]
        
        # Important: make sure that prob of dummy classes are exactly 0
        dummy_mask = torch.tensor([[(1 if i <= mask_idx else 0) for i in range(max(self.mask_index+1))] for mask_idx in self.mask_index], device=q_xs.device)
        dummy_mask = torch.ones_like(q_xs) * dummy_mask
        q_xs *= dummy_mask
        
        _x = self._sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        
        z_cat = copy_flag * x + (1 - copy_flag) * _x
        return copy_flag * x + (1 - copy_flag) * _x, q_xs

    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    
    def sample_ctime_noise(self, batch):
        if self.noise_dist == 'log_norm':
            rnd_normal = torch.randn(batch.shape[0], device=batch.device)
            sigma = (rnd_normal * self.noise_dist_params['P_std'] + self.noise_dist_params['P_mean']).exp()
        else:
            raise NotImplementedError(f"The noise distribution--{self.noise_dist}-- is not implemented for CTIME ")
        return sigma

    def _edm_loss(self, D_yn, y, sigma):
        weight = (sigma ** 2 + self.edm_params['sigma_data'] ** 2) / (sigma * self.edm_params['sigma_data']) ** 2
    
        target = y
        loss = weight * ((D_yn - target) ** 2)

        return loss
    
    def edm_update(
            self, x_num_cur, x_cat_cur, i, 
            t_cur, t_next, t_hat,
            sigma_num_cur, sigma_num_next, sigma_num_hat, 
            sigma_cat_cur, sigma_cat_next, sigma_cat_hat, 
            guidance_scale=0.0
        ):
        """
        i = T-1,...,0
        """
        cfg = self.y_only_model is not None
        
        b = x_num_cur.shape[0]
        has_cat = len(self.num_classes) > 0
        
        # Get x_num_hat by move towards the noise by a small step
        x_num_hat = x_num_cur + (sigma_num_hat ** 2 - sigma_num_cur ** 2).sqrt() * S_noise * torch.randn_like(x_num_cur)
        # Get x_cat_hat
        move_chance = -torch.expm1(sigma_cat_cur - sigma_cat_hat)    # the incremental move change is 1 - alpha_t/alpha_s = 1 - exp(sigma_s - sigma_t)
        x_cat_hat, _ = self.q_xt(x_cat_cur, move_chance) if has_cat else (x_cat_cur, x_cat_cur)

        # Get predictions
        x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_hat.dtype) if has_cat else x_cat_hat
        denoised_out = self._denoise_fn(
            x_num_hat.float(), x_cat_hat_oh,
            t_hat.squeeze().repeat(b), sigma=sigma_num_hat.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
        )
        if len(denoised_out) == 3:
             denoised, raw_logits, _ = denoised_out
        else:
             denoised, raw_logits = denoised_out

        # Negative Fairness Guidance
        if guidance_scale > 0 and getattr(self, 'fairness_enabled', False):
             # 1. Create Null input (Mask Sensitive Attributes)
             x_cat_hat_null = x_cat_hat.clone()
             if hasattr(self, 'sens_cat_idx'):
                 for idx in self.sens_cat_idx:
                     x_cat_hat_null[:, idx] = self.mask_index[idx] # Set to mask token
             
             x_cat_hat_oh_null = self.to_one_hot(x_cat_hat_null).to(x_num_hat.dtype)
             
             # 2. Forward pass for Null
             # Note: We reuse x_num_hat as is. 
             # (If we supported sensitive numericals, we should mask them too, but ASGDiff usually works with Cat sensitive)
             denoised_out_null = self._denoise_fn(
                 x_num_hat.float(), x_cat_hat_oh_null,
                 t_hat.squeeze().repeat(b), sigma=sigma_num_hat.unsqueeze(0).repeat(b,1)
             )
             if len(denoised_out_null) == 3:
                  denoised_null, raw_logits_null, _ = denoised_out_null
             else:
                  denoised_null, raw_logits_null = denoised_out_null
                  
             # 3. Compute Delta (Bias)
             # Delta = Cond - Null
             delta_num = denoised - denoised_null
             delta_cat_logits = raw_logits - raw_logits_null
             
             # Safety: Clamp Delta Num to avoid explosion
             # Bound numerical corrections in the standardized feature space.
             # Guidance scale 1.5 * 3 = 4.5 shift is huge.
             # We dampen the delta if it's too large.
             delta_num = torch.clamp(delta_num, -2.0, 2.0)
             
             # Safety: Clamp Delta Logits
             delta_cat_logits = torch.clamp(delta_cat_logits, -5.0, 5.0)

             # 4. Project Delta (Zero out sensitive cols to avoid modifying them)
             if len(self.sens_num_idx) > 0:
                 delta_num[:, self.sens_num_idx] = 0.0
             
             
             # ---------------------------------------------------------
             # DEEP FIX: Target-Only Guidance (Whitelist Approach)
             # ---------------------------------------------------------
             
             # 1. Numerical: Explicitly Zero out unless target exists
             safe_delta_num = torch.zeros_like(delta_num)
             if len(self.target_num_idx) > 0:
                 for idx in self.target_num_idx:
                     safe_delta_num[:, idx] = delta_num[:, idx]
             delta_num = safe_delta_num

             # 2. Categorical: Explicitly Zero out unless target exists
             safe_delta_cat = torch.zeros_like(delta_cat_logits)
             if len(self.target_cat_idx) > 0:
                 for idx in self.target_cat_idx:
                     # slices_for_classes_with_mask stores numpy arrays of indices
                     slice_indices = self.slices_for_classes_with_mask[idx]
                     # Map to start/end
                     start = slice_indices[0]
                     end = slice_indices[-1] + 1
                     safe_delta_cat[:, start:end] = delta_cat_logits[:, start:end]
             delta_cat_logits = safe_delta_cat
             
             # 3. Apply Clamping (Safety)
             delta_num = torch.clamp(delta_num, -2.0, 2.0)
             delta_cat_logits = torch.clamp(delta_cat_logits, -5.0, 5.0)

             # 4. Apply Guidance
             # Final = Cond - scale * Delta
             denoised = denoised - guidance_scale * delta_num
             raw_logits = raw_logits - guidance_scale * delta_cat_logits
             
             # 5. Final Safety Clamp
             denoised = torch.clamp(denoised, -5.0, 5.0)

        
        # Apply cfg updates, if is in cfg mode
        is_bin_class = len(self.num_mask_idx) == 0
        is_learnable = self.scheduler=="power_mean_per_column"
        if cfg:
            if not is_learnable:
                sigma_cond = sigma_num_hat
            else:
                if is_bin_class:
                    sigma_cond = (0.002 ** (1/7) + t_hat * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
                else:
                    sigma_cond = sigma_num_hat[self.num_mask_idx]
            y_num_hat = x_num_hat.float()[:, self.num_mask_idx]
            idx = list(chain(*[self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]))
            y_cat_hat = x_cat_hat_oh[:,idx]
            y_only_denoised, y_only_raw_logits = self.y_only_model(
                y_num_hat, 
                y_cat_hat,
                t_hat.squeeze().repeat(b), sigma=sigma_cond.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
            )
            
            denoised[:, self.num_mask_idx] *= 1 + self.w_num
            denoised[:, self.num_mask_idx] -= self.w_num*y_only_denoised
            
            mask_logit_idx = [self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]
            mask_logit_idx = np.concatenate(mask_logit_idx) if len(mask_logit_idx)>0 else np.array([])
            
            raw_logits[:, mask_logit_idx] *= 1 + self.w_cat
            raw_logits[:, mask_logit_idx] -= self.w_cat*y_only_raw_logits
        
        # Euler step
        d_cur = (x_num_hat - denoised) / sigma_num_hat
        x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * d_cur
        
        # Unmasking
        x_cat_next = x_cat_cur
        q_xs = torch.zeros_like(x_cat_cur).float()
        if has_cat:
            logits = self._subs_parameterization(raw_logits, x_cat_hat)
            alpha_t = torch.exp(-sigma_cat_hat).unsqueeze(0).repeat(b,1)
            alpha_s = torch.exp(-sigma_cat_next).unsqueeze(0).repeat(b,1)
            x_cat_next, q_xs = self._masked_diffusion_update(logits, x_cat_hat, alpha_t, alpha_s)
        
        # Apply 2nd order correction.
        if self.sampler_params['second_order_correction']:
            if i > 0:
                x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_next.dtype) if has_cat else x_cat_hat
                # If t_next is None (last step), we skip second order correction or use t=0 
                # Ideally, if i=0, we are at the end. 
                if t_next is not None:
                    t_next_val = t_next
                else:
                    t_next_val = torch.zeros_like(t_hat) # Use 0 time for last step correction
                
                denoised_out = self._denoise_fn(
                    x_num_next.float(), x_cat_hat_oh,
                    t_next_val.squeeze().repeat(b), sigma=sigma_num_next.unsqueeze(0).repeat(b,1)
                )
                if len(denoised_out) == 3:
                    denoised, raw_logits, _ = denoised_out
                else:
                    denoised, raw_logits = denoised_out

                if cfg:
                    if not is_learnable:
                        sigma_cond = sigma_num_next
                    else:
                        if is_bin_class:
                            sigma_cond = (0.002 ** (1/7) + t_next * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
                        else:
                            sigma_cond = sigma_num_next[self.num_mask_idx]
                    y_num_next = x_num_next.float()[:, self.num_mask_idx]
                    idx = list(chain(*[self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]))
                    y_cat_hat = x_cat_hat_oh[:, idx]
                    if t_next is not None:
                        t_next_val = t_next
                    else:
                        t_next_val = torch.zeros_like(t_hat)
                        
                    y_only_denoised, y_only_raw_logits = self.y_only_model(
                        y_num_next,
                        y_cat_hat,
                        t_next_val.squeeze().repeat(b), sigma=sigma_cond.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
                    )
                    denoised[:, self.num_mask_idx] *= 1 + self.w_num
                    denoised[:, self.num_mask_idx] -= self.w_num*y_only_denoised
                
                d_prime = (x_num_next - denoised) / sigma_num_next
                x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_num_next, x_cat_next, q_xs


    def sample_impute(self, x_num, x_cat, num_mask_idx, cat_mask_idx, resample_rounds, impute_condition, w_num, w_cat):
        self.w_num = w_num
        self.w_cat = w_cat
        self.num_mask_idx = num_mask_idx
        self.cat_mask_idx = cat_mask_idx
        
        b = x_num.size(0)
        device = self.device
        dtype = torch.float32

        # Create masks, true for the missing columns
        num_mask = [i in num_mask_idx for i in range(self.num_numerical_features)]
        cat_mask = [i in cat_mask_idx for i in range(len(self.num_classes))]
        num_mask = torch.tensor(num_mask).to(x_num.device).to(x_num.dtype)
        cat_mask = torch.tensor(cat_mask).to(x_cat.device).to(x_cat.dtype)

        # Create the chain of t
        t = torch.linspace(0,1,self.num_timesteps, dtype=dtype, device=device)      # times = 0.0,...,1.0
        t = t[:, None]
        
        # Compute the chains of sigma
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]
        
        # Prepare sigma_hat for stochastic sampling mode
        if self.sampler_params['stochastic_sampler']:
            gamma = min(S_churn / self.num_timesteps, np.sqrt(2) - 1) * (S_min <= sigma_num_cur) * (sigma_num_cur <= S_max)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(t_hat, dim=-1, keepdim=True).values    # take the samllest t_hat induced by sigma_num
            zero_gamma = (gamma==0).any()
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat = t
            sigma_num_hat = sigma_num_cur
            sigma_cat_hat = sigma_cat_cur

        # Sample priors for the continuous dimensions
        if impute_condition == "x_t":
            z_norm = x_num + torch.randn((b, self.num_numerical_features), device=device) * sigma_num_cur[-1]   # z_{t_max} = x_0(masked) + sigma_max*epsilon
        elif impute_condition == "x_0":
            z_norm = x_num
            
        # Sample priors for the discrete dimensions
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros((b, 0), device=device).float()      # the default values for categorical sample if the dataset has no categorical entry
        if has_cat:
            if impute_condition == "x_t":
                z_cat = self._sample_masked_prior(
                    b,
                    len(self.num_classes),
                )   # z_{t_max} is still all pushed to [MASK]
            elif impute_condition == "x_0":
                z_cat = x_cat
        
        pbar = tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps)
        pbar.set_description(f"Sampling Progress")
        for i in pbar:
            for u in range (resample_rounds):
                # Get known parts by Forward Flow
                if impute_condition == "x_t":
                    z_norm_known = x_num + torch.randn((b, self.num_numerical_features), device=device) * sigma_num_next[i]
                    move_chance = 1 - torch.exp(-sigma_cat_next[i]) if i < (self.num_timesteps-1) else torch.ones_like(sigma_cat_next[i])     # force move_chance to be 1 for the first iteration
                    z_cat_known, _ = self.q_xt(x_cat, move_chance)
                elif impute_condition == "x_0":
                    z_norm_known = x_num
                    z_cat_known = x_cat
                
                # Get unknown by Reverse Step
                z_norm_unknown, z_cat_unknown, q_xs = self.edm_update(
                    z_norm, z_cat, i, 
                    t[i], t[i-1] if i > 0 else None, t_hat[i],
                    sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i], 
                    sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
                )
                z_norm = (1 - num_mask)  * z_norm_known + num_mask * z_norm_unknown
                z_cat = (1 - cat_mask) * z_cat_known + cat_mask * z_cat_unknown

                # Resample x_t from x_{t-1} by Foward Step
                if u < resample_rounds-1:
                    z_norm = z_norm + (sigma_num_cur[i] ** 2 - sigma_num_next[i] ** 2).sqrt() * S_noise * torch.randn_like(z_norm)
                    move_chance = -torch.expm1(sigma_cat_next[i] - sigma_cat_cur[i])
                    z_cat, _ = self.q_xt(z_cat, move_chance)
        
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample
