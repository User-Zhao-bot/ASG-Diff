# ASG-Diff

ASG-Diff is a fairness-aware diffusion framework for mixed-type tabular data. It combines a mixed-type diffusion backbone with three lifecycle interventions:

- adversarial invariant bottleneck (AIB) during representation learning;
- Siamese counterfactual denoising (SCD) during training;
- selective guided null-condition mitigation (SGNM) during sampling.

This repository is a source-code release. It contains the model, training and sampling pipeline, dataset metadata, preprocessing utilities, and evaluation code. Pretrained checkpoints, raw or processed datasets, generated tables, experiment logs, and manuscript materials are not included.

## Repository layout

```text
.
|-- main.py                         # command-line entry point
|-- asgdiff/                        # model, diffusion, trainer, and configuration
|-- src/                            # data transformation and shared utilities
|-- eval/                           # fidelity and downstream-utility evaluation
|-- data/Info/                      # metadata only; no dataset records
|-- download_*.py                   # dataset acquisition utilities
|-- process_dataset.py              # preprocessing pipeline
|-- eval_fairness.py                # group-fairness evaluation
|-- asgdiff.yaml                    # primary Conda environment
`-- synthcity.yaml                  # optional fidelity-metric environment
```

## Environment

The reference environment uses Python 3.10 and PyTorch 2.0.1:

```bash
conda env create -f asgdiff.yaml
conda activate asgdiff
```

The optional SynthCity metrics use a separate environment because of dependency conflicts:

```bash
conda env create -f synthcity.yaml
```

## Dataset preparation

No dataset records are committed to this repository. The supplied utilities download publicly available sources; users remain responsible for reviewing each dataset's license and terms of use.

```bash
# Adult and Default (the downloader may also fetch additional supported datasets)
python download_dataset.py

# German Credit
python download_german.py

# COMPAS, Bank, LSAC, and MEPS
python download_fair_datasets.py --dataname all
```

Process the seven evaluation datasets individually:

```bash
python process_dataset.py --dataname adult
python process_dataset.py --dataname default
python process_dataset.py --dataname german
python process_dataset.py --dataname compas
python process_dataset.py --dataname bank
python process_dataset.py --dataname lsac
python process_dataset.py --dataname meps
```

The command-line sensitive indices refer to the categorical feature order before the binary target is internally prepended:

| Dataset | Sensitive field | `--sens_cat_idx` |
|---|---|---:|
| Adult | `sex` | 6 |
| Default | `SEX` | 0 |
| German | `Personal_status_sex` | 5 |
| COMPAS | `race` | 1 |
| Bank | `age_group` | 9 |
| LSAC | `race_bin` | 1 |
| MEPS | `RACE` | 2 |

## Training

The following example trains ASG-Diff on Adult with AIB and SCD enabled after the fairness warmup:

```bash
python main.py \
  --dataname adult \
  --mode train \
  --sens_cat_idx 6 \
  --lambda_adv 1.0 \
  --alpha_consistency 1.0 \
  --fairness_warmup_epochs 500 \
  --exp_name adult_asgdiff \
  --no_wandb
```

Checkpoints are written locally under `asgdiff/ckpt/` and are ignored by Git.

## Sampling

Sampling requires a checkpoint produced by the training command. SGNM is controlled by `--guidance_scale` and requires the same sensitive-field specification used for training.

```bash
python main.py \
  --dataname adult \
  --mode test \
  --ckpt_path asgdiff/ckpt/adult/adult_asgdiff/best_ema_model_<step>.pt \
  --sens_cat_idx 6 \
  --guidance_scale 1.5 \
  --num_samples_to_generate 32561 \
  --no_wandb
```

Set `--guidance_scale 0` for unguided sampling. Generated tables and evaluation outputs are excluded from version control.

## Evaluation

Group fairness can be evaluated from a real reference table and a generated table:

```bash
python eval_fairness.py \
  --real_data_path data/adult/adult.data \
  --fake_data_path path/to/samples.csv \
  --sensitive_col sex \
  --target_col income
```

The core pipeline also provides density, machine-learning efficacy, C2ST, and optional SynthCity metrics. Run `python main.py --help` for the current command-line interface.

## Released and withheld artifacts

The public repository intentionally excludes large or redistribution-sensitive artifacts: pretrained weights, optimizer states, raw and processed data, generated samples, W&B runs, internal experiment summaries, and paper-writing materials. Their absence is disclosed here rather than represented as a complete artifact bundle. See [RELEASE_SCOPE.md](RELEASE_SCOPE.md) for the exact boundary.

## License and attribution

The repository includes the MIT license notice required for redistributed components. The released implementation adds the fairness-aware training and sampling components described above.
