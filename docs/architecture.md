# Architecture Notes

## Why V-JEPA 2?

V-JEPA 2 (Meta FAIR, [June 2025](https://arxiv.org/abs/2506.09985)) is a self-supervised video model trained on 1M+ hours of internet video using a mask-denoising objective in representation space. It predicts *masked spatiotemporal patches in learned latent space*, not pixels.

**Why this matters for robotics:**

| Property | VLMs (CLIP, SigLIP) | V-JEPA 2 |
|---|---|---|
| Training signal | Image-text contrastive | Spatiotemporal prediction |
| Temporal reasoning | None (static frames) | Built-in |
| What it learns | "What is this?" | "What changes and how?" |
| Transfer to manipulation | Semantic features | Dynamics features |

The V-JEPA 2 paper (§4) already demonstrates zero-shot pick-and-place on Franka arms using only 62 hours of DROID post-training. This validates the representation transfer hypothesis — the encoder learns contact, deformation, and object motion implicitly.

**Key architectural details** (from the paper):
- ViT-L backbone (304M encoder params)
- Block-causal attention in the action-conditioned predictor
- EMA encoder for prediction targets (stable training)
- Progressive spatial/temporal resolution during pretraining

## Why SE(3) Flow Matching?

A robot gripper operates in SE(3) — the Lie group of rigid-body transformations. Standard action parameterizations ignore this structure:

| Method | Geometric Consistency | Rotation Validity | Interpolation |
|---|---|---|---|
| Direct regression (MLP) | None | Needs projection | Euclidean (wrong) |
| Diffusion (Euclidean) | None | Needs projection | Euclidean (wrong) |
| **Flow matching (SE(3))** | **Built-in** | **Always valid** | **Geodesic (correct)** |

SE(3) flow matching learns a continuous vector field on the manifold:

```
dφ_t/dt = v_θ(φ_t, t)    where φ_t ∈ SE(3), t ∈ [0, 1]
```

- **Translation**: 3D vector in ℝ³
- **Rotation**: 3D vector in 𝔰𝔬(3) (Lie algebra), mapped to SO(3) via exp map
- **Gripper**: Binary via sigmoid

This is grounded in recent work on flow matching for robotics ([FlowRAM, CVPR 2025](https://cvpr.thecvf.com/virtual/2025/poster/33579); [VFP, 2025](https://arxiv.org/abs/2508.01622)).

## Why Adaptive Neural ODE?

Standard networks use the same compute for every input. A simple "move forward" and a complex "pick, rotate, insert" pass through identical layers. Adaptive Neural ODEs make computation input-dependent:

```
dh/dt = f_θ(h, t, c)    where c = complexity embedding
```

The solver adaptively chooses function evaluations (NFEs) based on task complexity:

| Task | NFEs | Latency (A100) |
|---|---|---|
| Straight-line motion | 5-8 | ~5ms |
| Pick and place | 12-16 | ~10ms |
| Multi-step assembly | 20-30 | ~20ms |

This is conceptually related to early-exit and adaptive computation literature, applied to the action head rather than the backbone.

## Why Conformal Prediction?

Neural networks output point predictions with no guarantees. A model can be confidently wrong — dangerous in physical environments.

Conformal prediction provides **distribution-free, finite-sample coverage guarantees**:

> Under exchangeability, the true action falls within the predicted set with probability ≥ 1 - α.

This is not a heuristic. It's a mathematical guarantee used in robotics safety ([SAFE, 2025](https://arxiv.org/abs/2506.09937); [UNISafe, CMU](https://cmu-intentlab.github.io/UNISafe/)).

**How it works:**
1. Calibration: compute nonconformity scores on held-out data
2. Threshold: find `q_hat` such that `P(score ≤ q_hat) ≥ 1 - α`
3. Deploy: accept actions where `score ≤ q_hat`

| α | Coverage | Use case |
|---|---|---|
| 0.10 | ≥ 90% | Standard tasks |
| 0.05 | ≥ 95% | Careful manipulation |
| 0.01 | ≥ 99% | Safety-critical |

## Ablation Study Design

To understand each component's contribution:

| Variant | Backbone | Action Head | Neural ODE | Conformal | What It Tests |
|---|---|---|---|---|---|
| Full VL-JEPA | V-JEPA 2 | SE(3) FM | ✓ | ✓ | — |
| No Neural ODE | V-JEPA 2 | SE(3) FM | ✗ | ✓ | Compute adaptivity |
| No conformal | V-JEPA 2 | SE(3) FM | ✓ | ✗ | Safety filtering |
| MLP head | V-JEPA 2 | MLP | ✓ | ✓ | Flow matching benefit |
| CLIP backbone | CLIP ViT-L | SE(3) FM | ✓ | ✓ | V-JEPA 2 vs VLM features |

```bash
python scripts/run_ablation.py \
    --ablation_config configs/ablations/component.yaml \
    --num_seeds 5 \
    --output_dir results/ablations
```

---

Next: [Setup Guide](setup.md) | [Training Guide](training.md)
