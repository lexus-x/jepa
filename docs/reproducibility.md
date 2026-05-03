# Reproducibility Checklist

## Random Seed Handling

VL-JEPA sets seeds for all random sources:

```python
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Config: `seed: 42` in `configs/config.yaml`

## Deterministic Mode

For fully reproducible runs (slower):

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m src.training.trainer training.deterministic=true
```

This enables `torch.use_deterministic_algorithms(True)` which errors on non-deterministic ops.

## Environment Export

```bash
# Conda
conda env export --name vljepa > environment.yml
conda env export --name vljepa --no-builds > environment-nobuilds.yml

# Pip
pip freeze > requirements-frozen.txt
```

## Dependency Freezing

`requirements.txt` pins major dependencies to compatible ranges.
For exact reproduction, use `requirements-frozen.txt` generated from the exact training environment.

## Experiment Manifest

Each training run generates a manifest YAML:

```yaml
experiment:
  name: vljepa-libero-spatial
  timestamp: "2026-05-04T06:44:00+08:00"
  git_commit: abc1234
  git_dirty: false

config:
  model: vljepa
  training: default
  benchmark: libero
  seed: 42

environment:
  python: "3.10.14"
  torch: "2.3.0"
  cuda: "12.1"
  gpu: "NVIDIA A100 80GB"
  gpus: 8

results:
  libero_spatial: 0.962
  libero_object: 0.958
  libero_goal: 0.951
  libero_long: 0.925
  libero_avg: 0.949
```

## Metadata Logging

All experiments log to W&B with:
- Full config (Hydra structured config)
- Git commit hash and dirty status
- Environment details (Python, PyTorch, CUDA versions)
- GPU model and count
- Training metrics (loss, LR, grad norm)
- Evaluation metrics (success rates per task)
- Checkpoint paths

## Reproduction Steps

1. Clone repo at specific commit
2. Create conda environment from `environment.yml`
3. Download V-JEPA 2 checkpoints via `tools/download_checkpoints.sh`
4. Run training with exact config: `python -m src.training.trainer --config-name=config`
5. Compare results against manifest

## Checklist

- [ ] Seeds set for all random sources
- [ ] Deterministic mode enabled (optional, slower)
- [ ] Environment exported (`environment.yml`)
- [ ] Git commit recorded
- [ ] Config logged (full Hydra YAML)
- [ ] Checkpoints saved with metadata
- [ ] W&B run ID recorded
- [ ] Results compared against expected values
