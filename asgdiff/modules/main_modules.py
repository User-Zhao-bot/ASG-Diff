from typing import Callable, Union

from asgdiff.modules.transformer import Reconstructor, Tokenizer, Transformer
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Function

ModuleType = Union[str, Callable[..., nn.Module]]

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t = 512, use_mlp=True):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        ) if use_mlp else nn.Linear(dim_t, d_in)

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        
        self.use_mlp = use_mlp
    
    def forward(self, x, timesteps):
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)
    
        x = self.proj(x) + emb
        return self.mlp(x)

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)

    
class UniModMLP(nn.Module):
    """
        Input:
            x_num: [bs, d_numerical]
            x_cat: [bs, len(categories)]
        Output:
            x_num_pred: [bs, d_numerical], the predicted mean for numerical data
            x_cat_pred: [bs, sum(categories)], the predicted UNORMALIZED logits for categorical data
    """
    def __init__(
            self, d_numerical, categories, num_layers, d_token,
            n_head = 1, factor = 4, bias = True, dim_t=512, use_mlp=True, 
            sensitive_num_indices=[], sensitive_cat_indices=[], lambda_aib=1.0,
            **kwargs
        ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)
        self.encoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        d_in = d_token * (d_numerical + len(categories))
        self.mlp = MLPDiffusion(d_in, dim_t=dim_t, use_mlp=use_mlp)
        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)
        
        self.model = nn.ModuleList([self.tokenizer, self.encoder, self.mlp, self.decoder, self.detokenizer])

        # AIB Modules
        self.sensitive_num_indices = sensitive_num_indices
        self.sensitive_cat_indices = sensitive_cat_indices
        self.lambda_aib = lambda_aib
        
        # Calculate indices for slicing
        # Tokenizer output: [CLS, Num_1, ..., Num_N, Cat_1, ..., Cat_M]
        # Indices in e:
        # CLS: 0
        # Num: 1 to d_numerical
        # Cat: d_numerical + 1 to d_numerical + len(categories)
        
        self.sens_token_indices = []
        # Numerical
        for idx in sensitive_num_indices:
            self.sens_token_indices.append(1 + idx)
        # Categorical
        for idx in sensitive_cat_indices:
            self.sens_token_indices.append(1 + d_numerical + idx)
            
        self.sens_token_indices = sorted(list(set(self.sens_token_indices))) # Helper for extraction
        
        all_indices = set(range(1 + d_numerical + len(categories))) # Including CLS
        self.non_sens_token_indices = sorted(list(all_indices - set(self.sens_token_indices)))

        if len(self.sens_token_indices) > 0:
            # Flattened non-sensitive dimension
            dim_non_sens = len(self.non_sens_token_indices) * d_token
            
            # Discriminator output dimension
            # For numerical: 1 output per feature
            # For categorical: K outputs per feature (where K is cardinality)
            dim_out_sens = len(self.sensitive_num_indices)
            for idx in self.sensitive_cat_indices:
                dim_out_sens += self.categories[idx]
            
            self.grl = GradientReversalLayer(lambda_=lambda_aib)
            self.discriminator = Discriminator(dim_non_sens, dim_out_sens)
        else:
            self.grl = None
            self.discriminator = None


    def forward(self, x_num, x_cat, timesteps):
        e = self.tokenizer(x_num, x_cat)
        
        pred_sens = None
        if self.discriminator is not None:
             # AIB Logic
             # Split embeddings
             e_non_sens = e[:, self.non_sens_token_indices, :]
             
             # Flatten non-sensitive for discriminator
             e_non_sens_flat = e_non_sens.reshape(e_non_sens.size(0), -1)
             
             # Adversarial path
             pred_sens = self.discriminator(self.grl(e_non_sens_flat))
             
             # The reconstruction path and adversarial branch share the token
             # representation, so gradients from the GRL regularize the encoder.

        decoder_input = e[:, 1:, :]        # ignore the first CLS token. 
        y = self.encoder(decoder_input)
        pred_y = self.mlp(y.reshape(y.shape[0], -1), timesteps)
        pred_e = self.decoder(pred_y.reshape(*y.shape))
        x_num_pred, x_cat_pred = self.detokenizer(pred_e)
        x_cat_pred = torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred)>0 else torch.zeros_like(x_cat).to(x_num_pred.dtype)

        return x_num_pred, x_cat_pred, pred_sens



class Precond(nn.Module):
    def __init__(self,
        denoise_fn,
        sigma_data = 0.5,              # Expected standard deviation of the training data.
        net_conditioning = "sigma",
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.net_conditioning = net_conditioning
        self.denoise_fn_F = denoise_fn

    def forward(self, x_num, x_cat, t, sigma):

        x_num = x_num.to(torch.float32)

        sigma = sigma.to(torch.float32)
        assert sigma.ndim == 2
        if sigma.dim() > 1: # if learnable column-wise noise schedule, sigma conditioning is set to the defaults schedule of rho=7
            sigma_cond = (0.002 ** (1/7) + t * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
        else:
            sigma_cond = sigma 
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma_cond.log() / 4

        x_in = c_in * x_num
        if self.net_conditioning == "sigma":
            out = self.denoise_fn_F(x_in, x_cat, c_noise.flatten())
        elif self.net_conditioning == "t":
            out = self.denoise_fn_F(x_in, x_cat, t)
            
        if len(out) == 3:
            F_x, x_cat_pred, pred_sens = out
        else:
            F_x, x_cat_pred = out
            pred_sens = None

        assert F_x.dtype == dtype
        D_x = c_skip * x_num + c_out * F_x.to(torch.float32)
        
        return D_x, x_cat_pred, pred_sens

    

class Model(nn.Module):
    def __init__(
            self, denoise_fn,
            sigma_data=0.5, 
            precond=False, 
            net_conditioning="sigma",
            **kwargs
        ):
        super().__init__()
        self.precond = precond
        if precond:
            self.denoise_fn_D = Precond(
                denoise_fn,
                sigma_data=sigma_data,
                net_conditioning=net_conditioning
            )
        else:
            self.denoise_fn_D = denoise_fn

    def forward(self, x_num, x_cat, t, sigma=None):
        if self.precond:
            return self.denoise_fn_D(x_num, x_cat, t, sigma)
        else:
            return self.denoise_fn_D(x_num, x_cat, t)

