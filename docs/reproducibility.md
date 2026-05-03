# Reproducibility

## Random Seeds

Single `--seed` flag controls all sources:

```bash
python scripts/train.py --seed 42
```

Sets: `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`, cuDNN settings, DataLoader worker seeds.

For publication results, run multiple seeds:

```bash
for seed in 42 43 44 45 46; do
    torchrun --nproc_per_node=4 scripts/train.py \
        --config configs/phase4_finetune/libero_spatial.yaml \
        --seed $seed \
        --output_dir outputs/seed_${seed}
done
```

Report mean ± std.

## Deterministic Mode

For exact reproducibility (slower, ~10-15% overhead):

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

python scripts/train.py --deterministic --seed 42
```

Or in config:

```yaml
training:
  deterministic: true
  seed: 42
```

| Setting | Normal | Deterministic |
|---|---|---|
| cuDNN benchmark | Enabled | Disabled |
| cuDNN deterministic | Off | On |
| `use_deterministic_algorithms` | Off | On |

## Environment Export

```bash
# Full conda export (run at experiment start)
conda env export --no-builds > environment.yml
pip freeze > requirements_frozen.txt
```

Reconstruct later:

```bash
conda env create -f environment.yml
```

## Experiment Manifest

Each run auto-generates `outputs/<run>/manifest.yaml`:

```yaml
experiment:
  name: phase2_posttrain_droid
  created_at: "2025-01-15T14:30:00Z"

model:
  architecture: vljepa
  backbone: vjepa2_vitl16
  checkpoint: checkpoints/vitl16_vjepa2.pt

training:
  phase: 2
  seed: 42
  deterministic: false

git:
  commit: "abc123"
  branch: "main"
  dirty: false
```

Generate for existing runs:

```bash
python scripts/generate_manifest.py \
    --run_dir outputs/phase2 \
    --config configs/phase2_posttrain.yaml
```

## Reproducing a Published Result

```bash
# 1. Clone at exact commit
git clone https://github.com/lexus-x/jepa.git
cd jepa
git checkout <commit_hash>

# 2. Recreate environment
conda env create -f environment.yml

# 3. Download checkpoints
bash scripts/download_vjepa2.sh

# 4. Run with exact seed + deterministic
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/phase2_posttrain.yaml \
    --seed 42 --deterministic \
    --output_dir outputs/reproduce
```

---

Next: [Evaluation Guide](evaluation.md) | [Contributing](../CONTRIBUTING.md)
