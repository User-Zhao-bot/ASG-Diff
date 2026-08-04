import glob
import json
import os
import pickle
import random
import sys

# Fix for MKL threading issue (Pitfall #7)
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import numpy as np
from asgdiff.metrics import TabMetrics
from asgdiff.modules.main_modules import UniModMLP
from asgdiff.modules.main_modules import Model
from asgdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from asgdiff.trainer import Trainer
import src
import torch

from torch.utils.data import DataLoader
import argparse
import warnings

from copy import deepcopy
from pathlib import Path

from utils_train import ASGDiffDataset

warnings.filterwarnings('ignore')


def configure_utf8_stdio() -> None:
    """Avoid Windows console encoding crashes inside metric reporters."""
    if os.name != 'nt':
        return
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(args):
    configure_utf8_stdio()
    device = args.device

    ## Disable scientific numerical format
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    ## Get data info
    dataname = args.dataname
    data_dir = f'data/{dataname}'
    info_path = f'data/{dataname}/info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    ## Set up flags
    is_dcr = 'dcr' in dataname

    ## Set experiment name
    exp_name = args.exp_name
    if args.exp_name is None:
        exp_name = 'non_learnable_schedule' if args.non_learnable_schedule else 'learnable_schedule'
    exp_name += '_y_only' if args.y_only else ''
    
    ## Load configs
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{curr_dir}/configs/asgdiff_configs.toml'
    raw_config = src.load_config(config_path)

    if args.mode == 'test' and args.guidance_scale != 0 and not (args.sens_num_idx or args.sens_cat_idx):
        raise ValueError(
            "Fairness-aware test requires explicit sensitive indices. "
            "Pass --sens_cat_idx/--sens_num_idx in the command. "
            "If you intend an unguided test, set --guidance_scale 0."
        )
    
    print(f"{args.mode.capitalize()} Mode is Enabled")
    num_samples_to_generate = args.num_samples_to_generate
    ckpt_path = None
    if args.mode == 'train':
        print("NEW training is started" if args.ckpt_path is None else f"Resuming training from {args.ckpt_path}")
        ckpt_path = args.ckpt_path
    elif args.mode == 'test':
        num_samples_to_generate = args.num_samples_to_generate
        ckpt_path = args.ckpt_path
        if ckpt_path is None:
            ckpt_parent_path = f"{curr_dir}/ckpt/{dataname}/{exp_name}"
            ckpt_path_arr = glob.glob(f"{ckpt_parent_path}/best_ema_model*")
            assert ckpt_path_arr, f"Cannot infer a checkpoint from {ckpt_parent_path}; train a model or pass --ckpt_path explicitly."
            ckpt_path = ckpt_path_arr[0]
        config_path = os.path.join(os.path.dirname(ckpt_path), 'config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                cached_raw_config = pickle.load(f)
                print(f"Found cached config at {config_path}")
        raw_config = cached_raw_config
    
    
    ## Create model and result paths
    model_save_path, result_save_path = None, None
    if args.mode == 'train':
        model_save_path = 'debug/ckpt' if args.debug else f'{curr_dir}/ckpt/{dataname}/{exp_name}'
        result_save_path = model_save_path.replace('ckpt', 'result')  #i.e., f'{curr_dir}/results/{dataname}/{exp_name}'
    elif args.mode == 'test':
        if args.report:
            result_save_path = f"eval/report_runs/{exp_name}/{dataname}"
        else:
            result_save_path = os.path.dirname(ckpt_path).replace('ckpt', 'result')    # infer the exp_name from the ckpt_name
        if args.result_subdir:
            result_save_path = str(Path(result_save_path) / args.result_subdir)
    raw_config['model_save_path'] = model_save_path
    raw_config['result_save_path'] = result_save_path
    if model_save_path is not None:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    if result_save_path is not None:
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)
    
    ## Set random seeds
    raw_config['seed'] = args.seed
    print(f"Using random seed: {args.seed}")
    set_global_seed(args.seed)

    ## Make everything determinstic if needed
    raw_config['deterministic'] = args.deterministic
    if args.deterministic:
        print("DETERMINISTIC MODE is enabled!!!")
        ## Ensure deterministic CUDA operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    ## Apply runtime config overrides
    if args.steps is not None:
        raw_config['train']['main']['steps'] = args.steps
    if args.check_val_every is not None:
        raw_config['train']['main']['check_val_every'] = args.check_val_every
    if args.batch_size is not None:
        raw_config['train']['main']['batch_size'] = args.batch_size
    if args.sample_batch_size is not None:
        raw_config['sample']['batch_size'] = args.sample_batch_size

    ## Set debug mode parameters
    if args.debug:  # fast eval for DEBUG mode
        raw_config['train']['main']['check_val_every'] = 2
        raw_config['diffusion_params']['num_timesteps'] = 4
        raw_config['train']['main']['batch_size'] = 4096
        raw_config['sample']['batch_size'] = 10000

    ## Load training data
    batch_size = raw_config['train']['main']['batch_size']

    train_data = ASGDiffDataset(dataname, data_dir, info, y_only=args.y_only, isTrain=True, dequant_dist=raw_config['data']['dequant_dist'], int_dequant_factor=raw_config['data']['int_dequant_factor'])
    train_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
    }
    if torch.cuda.is_available():
        train_loader_kwargs['pin_memory'] = True
    if args.num_workers > 0:
        train_loader_kwargs['persistent_workers'] = True
        train_loader_kwargs['prefetch_factor'] = 4
    train_loader = DataLoader(train_data, **train_loader_kwargs)
    d_numerical, categories = train_data.d_numerical, train_data.categories
    
    val_data = ASGDiffDataset(dataname, data_dir, info, y_only=args.y_only, isTrain=False, dequant_dist=raw_config['data']['dequant_dist'], int_dequant_factor=raw_config['data']['int_dequant_factor'])

    ## Load Metrics
    real_data_path = f'synthetic/{dataname}/real.csv'
    test_data_path = f'synthetic/{dataname}/test.csv'
    val_data_path = f'synthetic/{dataname}/val.csv'
    if not os.path.exists(val_data_path):
        print(f"{args.dataname} does not have its validation set. During MLE evaluation, a validation set will be splitted from the training set!")
        val_data_path = None
    if args.mode == 'train':
        metric_list = ["density"]
    else:
        if is_dcr:
            metric_list = ["dcr"]
        else:
            metric_list = [
                "density", 
                "mle", 
                "c2st",
            ]
    metrics = TabMetrics(real_data_path, test_data_path, val_data_path, info, device, metric_list=metric_list)
    
    ## Load the module and models
    raw_config['unimodmlp_params']['d_numerical'] = d_numerical
    raw_config['unimodmlp_params']['categories'] = (categories+1).tolist()  # add one for the mask category
    if args.y_only:
        raw_config['unimodmlp_params']['use_mlp'] = False     # drop the mlp when training the unconditional model
        raw_config['unimodmlp_params']['dim_t'] = 128   #reduce the size of the mlp
        main_model_path = args.ckpt_path
        if main_model_path is None:
            main_model_parent_path = f"{curr_dir}/ckpt/{dataname}/{exp_name.replace('_y_only', '')}"
            main_model_path_arr = glob.glob(f"{main_model_parent_path}/best_ema_model*")
            assert main_model_path_arr, f"Cannot infer a main-model checkpoint from {main_model_parent_path}; train the main model first or pass its path explicitly."
            main_model_path = main_model_path_arr[0]
        main_model_configs = pickle.load(open(os.path.join(os.path.dirname(main_model_path), 'config.pkl'), 'rb'))
        if main_model_configs['diffusion_params']['scheduler'] == "power_mean_per_column": # if learnable schedule is enabled in the main model, we need to infer noise params of the target column from the main model ckpt and train the y_only model with those params
            from asgdiff.models.noise_schedule import PowerMeanNoise_PerColumn, LogLinearNoise_PerColumn
            if info['task_type'] == 'regression':
                noise_schedule = PowerMeanNoise_PerColumn(
                    num_numerical=main_model_configs['unimodmlp_params']['d_numerical'], 
                    **main_model_configs['diffusion_params']['noise_schedule_params']
                )
                raw_config['diffusion_params']['noise_schedule_params']['rho'] = noise_schedule.rho()[0].item()    # the target col is placed at the first position
            else:
                noise_schedule = LogLinearNoise_PerColumn(
                    num_categories=len(main_model_configs['unimodmlp_params']['categories']), 
                    **main_model_configs['diffusion_params']['noise_schedule_params']
                )
                raw_config['diffusion_params']['noise_schedule_params']['k'] = noise_schedule.k()[0].item()    # the target col is placed at the first position
            
    
    # Pre-process Sensitive Indices for Backbone Initialization
    # We must do this BEFORE creating UniModMLP so it knows to create the Discriminator
    if args.sens_num_idx or args.sens_cat_idx:
        for idx in args.sens_num_idx:
            if idx < 0 or idx >= d_numerical:
                raise ValueError(
                    f"Invalid --sens_num_idx {idx} for dataset '{dataname}'. "
                    f"Expected numerical sensitive indices within [0, {d_numerical - 1}]."
                )

         # Also shift sens_cat_idx by 1 if y is prepended
        if len(categories) == len(info['cat_col_idx']) + 1:
             args.sens_cat_idx = [idx + 1 for idx in args.sens_cat_idx]

        for idx in args.sens_cat_idx:
            if idx < 0 or idx >= len(categories):
                raise ValueError(
                    f"Invalid --sens_cat_idx {idx} for dataset '{dataname}' after internal target shift. "
                    f"Expected categorical sensitive indices within [0, {len(categories) - 1}]."
                )

        raw_config['unimodmlp_params']['sensitive_num_indices'] = args.sens_num_idx
        raw_config['unimodmlp_params']['sensitive_cat_indices'] = args.sens_cat_idx
        raw_config['unimodmlp_params']['lambda_aib'] = args.lambda_adv

    backbone = UniModMLP(
        **raw_config['unimodmlp_params']
    )
    model = Model(backbone, **raw_config['diffusion_params']['edm_params'])
    model.to(device)
    
    ## Create and load y_only_model for imputation
    y_only_model = None
    if args.impute:
        y_only_model_path = args.y_only_model_path
        if y_only_model_path is None:
            y_only_model_parent_path = f"{curr_dir}/ckpt/{dataname}/{exp_name}_y_only"
            y_only_model_path_arr = glob.glob(f"{y_only_model_parent_path}/best_ema_model*")
            assert y_only_model_path_arr, f"Cannot infer a y-only checkpoint from {y_only_model_parent_path}; train the y-only model or pass --y_only_model_path explicitly."
            y_only_model_path = y_only_model_path_arr[0]
        y_only_model_config_path = os.path.join(os.path.dirname(y_only_model_path), 'config.pkl')
        with open(y_only_model_config_path, 'rb') as f:
                y_only_model_config = pickle.load(f)
        y_only_model = UniModMLP(
            **y_only_model_config['unimodmlp_params']
        )
        y_only_model = Model(y_only_model, **y_only_model_config['diffusion_params']['edm_params'])
        y_only_model.to(device)
        # load weights
        state_dicts = torch.load(y_only_model_path, map_location=device)
        y_only_model.load_state_dict(state_dicts['denoise_fn'])

    if not args.y_only and not args.non_learnable_schedule:
        raw_config['diffusion_params']['scheduler'] = 'power_mean_per_column'
        raw_config['diffusion_params']['cat_scheduler'] = 'log_linear_per_column'
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=y_only_model,
        **raw_config['diffusion_params'],
        device=device,
    )
    num_params = sum(p.numel() for p in diffusion.parameters())
    print("The number of parameters = ", num_params)
    diffusion.to(device)
    
    # Configure Fairness (SCD)
    if args.sens_num_idx or args.sens_cat_idx:
        print(f"Fairness Enabled: Sens Num {args.sens_num_idx}, Sens Cat {args.sens_cat_idx}")
        
        # Map global target columns to their positions in the model input.
        target_num_idx = []
        target_cat_idx = []
        target_col_idx = info['target_col_idx']

        for tgt_idx in target_col_idx:
            if tgt_idx < d_numerical:
                target_num_idx.append(tgt_idx)
            else:
                # Classification targets are prepended to the categorical block.
                if len(categories) == len(info['cat_col_idx']) + 1:
                     target_cat_idx.append(0)
                else:
                     target_cat_idx.append(tgt_idx - d_numerical)

        target_col_names = [info['column_names'][idx] for idx in target_col_idx]
        print(
            "Dataset fairness mapping: "
            f"target columns {target_col_names} from info.json (global {target_col_idx}) -> "
            f"relative target num {target_num_idx}, target cat {target_cat_idx}"
        )

        # NOTE: args.sens_cat_idx was already shifted above before backbone init

        if target_num_idx or target_cat_idx:
             print(f"Excluding Targets from Consistency: Num {target_num_idx}, Cat {target_cat_idx}")

        diffusion.set_fairness_config(
            sens_num_idx=args.sens_num_idx,
            sens_cat_idx=args.sens_cat_idx,
            target_num_idx=target_num_idx,
            target_cat_idx=target_cat_idx,
            lambda_adv=args.lambda_adv,
            alpha_consistency=args.alpha_consistency
        )
    
    diffusion.train()

    ## Print the configs
    printed_configs = json.dumps(raw_config, default=lambda x: int(x) if isinstance(x, np.int64) else x, indent=4)
    print(f"The config of the current run is : \n {printed_configs}")
    
    ## Enable Wandb
    project_name = f"asgdiff_{dataname}"
    raw_config['project_name'] = project_name
    # logger = wandb.init(
    #     project=raw_config['project_name'], 
    #     name=exp_name,
    #     config=raw_config,
    #     mode='disabled' if args.debug or args.no_wandb else 'online',
    # )
    class DummyLogger:
        def log(self, d):
            pass
        def define_metric(self, *args, **kwargs):
            pass
        def watch(self, *args, **kwargs):
            pass
    logger = DummyLogger()

    ## Load Trainer
    sample_batch_size = raw_config['sample']['batch_size']
    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        val_data,
        metrics,
        logger,
        **raw_config['train']['main'],
        sample_batch_size=sample_batch_size,
        num_samples_to_generate=num_samples_to_generate,
        model_save_path=raw_config['model_save_path'],
        result_save_path=raw_config['result_save_path'],
        device=device,
        ckpt_path=ckpt_path,
        y_only=args.y_only,
        aux_lambda=1.0, # Trainer expects aux_lambda to scale the aux loss if needed, 
                       # but mixed_loss already scales components. 
                       # We set aux_lambda=1.0 here so it adds the returned aux_loss directly.
        guidance_scale=args.guidance_scale,
        fairness_warmup_epochs=args.fairness_warmup_epochs
    )
    if args.mode == 'test':
        if args.report:
            if  is_dcr:
                trainer.report_test_dcr(args.num_runs)
            else:
                trainer.report_test(args.num_runs)
        elif args.impute:
            imputed_sample_save_dir = f"impute/{dataname}/{exp_name}"
            trainer.test_impute(
                args.trial_start, args.trial_size, 
                args.resample_rounds, 
                args.impute_condition, 
                imputed_sample_save_dir,
                args.w_num,
                args.w_cat,
            )
        else:
            trainer.test()
    else:
        ## Save config
        config_save_path = raw_config['model_save_path']
        with open (os.path.join(config_save_path, 'config.pkl'), 'wb') as f:
            pickle.dump(raw_config, f)
        trainer.run_loop()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training and sampling with ASG-Diff')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    
    # Fairness Arguments
    parser.add_argument('--sens_num_idx', nargs='+', type=int, default=[], help='Indices of sensitive numerical features')
    parser.add_argument('--sens_cat_idx', nargs='+', type=int, default=[], help='Indices of sensitive categorical features')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Weight for Adversarial Loss (AIB)')
    parser.add_argument('--alpha_consistency', type=float, default=1.0, help='Weight for Consistency Loss (SCD)')
    parser.add_argument('--guidance_scale', type=float, default=1.5, help='Scale for SGNM sampling guidance')
    parser.add_argument('--fairness_warmup_epochs', type=int, default=500, help='Epochs before enabling fairness loss')

    # General Config
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--y_only', action='store_true', help='Train y_only model')
    parser.add_argument('--non_learnable_schedule', action='store_true', help='Use non-learnable schedule')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--deterministic', action='store_true', help='Deterministic mode')
    parser.add_argument('--seed', type=int, default=0, help='Random seed used for numpy, random, torch, and cuda.')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--steps', type=int, default=None, help='Override training steps')
    parser.add_argument('--check_val_every', type=int, default=None, help='Override validation frequency in epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override training batch size')
    parser.add_argument('--sample_batch_size', type=int, default=None, help='Override sampling batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (set 0 on Windows for stability)')
    
    # Test/Impute Config
    parser.add_argument('--num_samples_to_generate', type=int, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--result_subdir', type=str, default=None, help='Optional subdirectory appended to test-time result outputs.')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--impute', action='store_true')
    parser.add_argument('--y_only_model_path', type=str, default=None)
    parser.add_argument('--num_runs', type=int, default=5)
    
    # Impute specific
    parser.add_argument('--trial_start', type=int, default=0)
    parser.add_argument('--trial_size', type=int, default=10)
    parser.add_argument('--resample_rounds', type=int, default=5)
    parser.add_argument('--impute_condition', type=str, default='x_0')
    parser.add_argument('--w_num', type=float, default=0.0)
    parser.add_argument('--w_cat', type=float, default=0.0)

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    main(args)
