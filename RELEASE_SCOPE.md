# Release scope

This repository is a source-code release, not a frozen reproduction artifact.

## Included

- ASG-Diff model and mixed-type diffusion backbone;
- AIB, SCD, and SGNM implementation;
- training, sampling, preprocessing, and evaluation entry points;
- model and dataset metadata configurations;
- dataset download utilities for publicly available sources;
- environment specifications and upstream license notice.

## Not included

- pretrained or intermediate checkpoints;
- optimizer states and serialized run configurations;
- raw, processed, or synthetic datasets;
- generated samples and metric outputs;
- W&B runs, logs, caches, and local development files;
- paper source, figures, review notes, and internal analysis scripts.

The excluded files are either large, generated, environment-specific, redistribution-sensitive, or part of the internal research workspace. Users can train new checkpoints from the released source after preparing the datasets under their applicable licenses. Exact numerical reproduction of a particular internal checkpoint is not claimed without that checkpoint and its associated run artifacts.
