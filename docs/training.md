# Training Guide

VL-JEPA uses a 4-phase pipeline. Each phase builds on the previous.

## Overview

| Phase | What | Data | Output |
|---|---|---|---|
| 1 | Use V-JEPA 2 pretrained encoder | Meta's checkpoint (1M+ hrs video) | Frozen backbone |
| 2 | Action-conditioned post-training | 62h DROID (robot trajectories) | Trained predictor |
| 3 | Conformal calibration | Held-out calibration split | Safety thresholds |
| 4 | Benchmark fine-tuning | LIBERO / MetaWorld task data | Task-specific model |

Phase 1 uses Meta's pretrained V-JEPA 2 directly — no retraining. See [setup.md](setup.md) for download.

## Phase 2: Action-Conditioned Post-Training

This is the main training phase. Following V-JEPA 2-AC ([paper §3](https://arxiv.org/html/2506.09985v1)), we train an action-conditioned predictor on top of the frozen V-JEPA 2 encoder using DROID robot trajectories.

### Data Preparation

```bash
# Download DROID dataset
python scripts/download_droid.py --output_dir data/droid --num_workers 8

# Preprocess (extract frames, compute SE(3) action labels)
python scripts/preprocess_droid.py \
    --input_dir data/droid \
    --output_dir data/droid_processed \
    --num_workers 16
```

### Single-GPU

```bash
python scripts/train.py \
    --config configs/phase2_posttrain.yaml \
    --data_path data/droid_processed \
    --output_dir outputs/phase2 \
    --seed 42
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/phase2_posttrain.yaml \
    --data_path data/droid_processed \
    --output_dir outputs/phase2 \
    --seed 42
```

### Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| V-JEPA 2 encoder | **Frozen** | Only predictor trains |
| Predictor params | ~300M | Block-causal transformer |
| Batch size / GPU | 16 | |
| Learning rate | 1e-4 | Cosine schedule, 1000 warmup steps |
| Epochs | 50 | |
| Image resolution | 224×224 | |
| Frame stack | 4 | |

## Phase 3: Conformal Calibration

Compute conformal prediction thresholds on held-out data. This gives provable coverage guarantees at deployment.

```bash
python scripts/calibrate_conformal.py \
    --checkpoint outputs/phase2/checkpoints/best.pt \
    --calibration_data data/droid_processed/calibration \
    --alpha 0.05 \
    --output_dir outputs/phase3
```

Produces `outputs/phase3/conformal_params.json` with `q_hat` (threshold) and coverage statistics.

## Phase 4: Benchmark Fine-Tuning

### LIBERO

```bash
# Each suite separately
for suite in spatial object goal long; do
    torchrun --nproc_per_node=4 scripts/train.py \
        --config configs/phase4_finetune/libero_${suite}.yaml \
        --pretrained outputs/phase2/checkpoints/best.pt \
        --conformal_params outputs/phase3/conformal_params.json \
        --data_path data/libero_${suite} \
        --output_dir outputs/phase4/libero_${suite} \
        --seed 42
done
```

**Note**: LIBERO eval uses a separate conda env (`libero`, Python 3.8). See [setup.md](setup.md).

### MetaWorld

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/phase4_finetune/metaworld_mt10.yaml \
    --pretrained outputs/phase2/checkpoints/best.pt \
    --data_path data/metaworld_mt10 \
    --output_dir outputs/phase4/metaworld_mt10 \
    --seed 42
```

## Experiment Tracking (W&B)

```yaml
# In config:
wandb:
  enabled: true
  project: vl-jepa
  entity: your-org
  tags: ["phase2", "droid"]
```

```bash
# Login
wandb login YOUR_API_KEY

# Override from CLI
python scripts/train.py --config configs/phase2_posttrain.yaml \
    --wandb_name phase2_run1 --wandb_tags "phase2,droid"
```

## Resuming

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/phase2_posttrain.yaml \
    --resume outputs/phase2/checkpoints/latest.pt
```

## Checkpoint Structure

```
outputs/<run_name>/
├── checkpoints/
│   ├── best.pt
│   ├── latest.pt
│   └── epoch_NNNN.pt
├── logs/train.log
├── config_snapshot.yaml
└── manifest.yaml          # Auto-generated metadata
```

---

Next: [Evaluation Guide](evaluation.md) | [Architecture Notes](architecture.md)
