# Evaluation Guide

## LIBERO

LIBERO has 4 task suites. Use the **separate** `libero` conda env (Python 3.8, MuJoCo).

```bash
conda activate libero
```

### Single Suite

```bash
python scripts/eval_benchmark.py \
    --benchmark libero_spatial \
    --checkpoint outputs/phase4/libero_spatial/checkpoints/best.pt \
    --conformal_params outputs/phase3/conformal_params.json \
    --num_episodes 50 \
    --output_dir results/libero_spatial
```

### All Suites

```bash
for suite in spatial object goal long; do
    python scripts/eval_benchmark.py \
        --benchmark libero_${suite} \
        --checkpoint outputs/phase4/libero_${suite}/checkpoints/best.pt \
        --conformal_params outputs/phase3/conformal_params.json \
        --num_episodes 50 \
        --output_dir results/libero_${suite}
done
```

### Aggregate

```bash
python scripts/aggregate_results.py --results_dir results/ --benchmark libero
```

### LIBERO Context

LIBERO is [essentially saturated](https://mbreuss.github.io/blog_post_iclr_26_vla.html). Rule of thumb for what "good" looks like:

| Suite | Competitive | Strong | Ceiling |
|---|---|---|---|
| Spatial | >95% | >97% | ~99% |
| Object | >95% | >97% | ~99% |
| Goal | >95% | >97% | ~99% |
| Long | >90% | >95% | ~97% |

A properly tuned Diffusion Policy can hit these without VLMs. The real differentiator is CALVIN and real-world transfer.

## MetaWorld

```bash
conda activate vljepa  # back to main env

# MT10
python scripts/eval_benchmark.py \
    --benchmark metaworld_mt10 \
    --checkpoint outputs/phase4/metaworld_mt10/checkpoints/best.pt \
    --num_episodes 100 \
    --output_dir results/metaworld_mt10

# MT50
python scripts/eval_benchmark.py \
    --benchmark metaworld_mt50 \
    --checkpoint outputs/phase4/metaworld_mt50/checkpoints/best.pt \
    --num_episodes 50 \
    --output_dir results/metaworld_mt50
```

## CALVIN

CALVIN ABC→D measures average task completion length (out of 5 subtasks). Current SOTA is ~4.44 ([DreamVLA](https://arxiv.org/abs/2507.04447), [Seer](https://arxiv.org/abs/2412.15109)).

```bash
python scripts/eval_benchmark.py \
    --benchmark calvin_abc \
    --checkpoint outputs/phase2/checkpoints/best.pt \
    --num_episodes 1000 \
    --output_dir results/calvin
```

## Metrics

| Metric | Formula | Use |
|---|---|---|
| Success rate | `successes / episodes` | LIBERO, MetaWorld |
| Avg completion length | `sum(subtasks) / episodes` | CALVIN |
| Conformal coverage | `(scores ≤ q_hat).mean()` | Safety (target: ≥ 1-α) |

## Visualization

```bash
# Single episode with overlay
python scripts/visualize_episode.py \
    --checkpoint outputs/phase4/libero_spatial/checkpoints/best.pt \
    --task "pick up the red block" \
    --output_dir visualizations/
```

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| Low success rate | Wrong checkpoint phase | Use Phase 4 checkpoint, not Phase 2 |
| `KeyError: conformal_params` | Missing calibration | Run Phase 3 first |
| MuJoCo rendering fails | Headless server | `export MUJOCO_GL=egl` |
| LIBERO import error | Wrong env | `conda activate libero` (Python 3.8) |
| GPU OOM during eval | Too many workers | Reduce `--num_workers` |

---

Next: [Reproducibility](reproducibility.md) | [Training Guide](training.md)
