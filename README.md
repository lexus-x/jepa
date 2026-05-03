# VL-JEPA: Vision-Language Joint-Embedding Predictive Architecture for Robot Manipulation

<p align="center">
  <a href="https://arxiv.org/abs/2506.09985"><img src="https://img.shields.io/badge/V--JEPA%202-Reference-blue" alt="V-JEPA 2"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" alt="PyTorch"></a>
</p>

> **Status: Research prototype.** No benchmark results yet — target numbers below are aspirational baselines from related work, not claims.

---

## What Is This?

VL-JEPA extends [V-JEPA 2](https://arxiv.org/abs/2506.09985) (Meta FAIR, June 2025) for closed-loop robot manipulation. The core hypothesis: a video model trained on 1M+ hours of internet video learns dynamics-aware representations that transfer better to robotics than VLM features trained for semantic alignment.

V-JEPA 2 already demonstrates zero-shot pick-and-place on Franka arms with only 62 hours of DROID post-training ([paper §4](https://arxiv.org/html/2506.09985v1)). VL-JEPA adds:

- **Language conditioning** for instruction-following (V-JEPA 2 is action-free by default)
- **SE(3) flow matching** head for geometrically consistent action generation
- **Adaptive Neural ODE** for variable-compute inference
- **Conformal prediction** for calibrated safety filtering

```
┌─────────────────────────────────────────────────────────────────┐
│                         VL-JEPA                                 │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │  RGB     │   │  V-JEPA 2    │   │  Adaptive Neural ODE   │  │
│  │  Frames  │──▶│  ViT-L       │──▶│  (complexity-adaptive) │  │
│  │  (T=4)   │   │  (frozen)    │   │                        │  │
│  └──────────┘   └──────┬───────┘   └───────────┬────────────┘  │
│                        │                       │                │
│                        ▼                       ▼                │
│                  ┌──────────┐        ┌──────────────────┐       │
│                  │  T5-XXS  │        │  SE(3) Flow      │       │
│                  │  Language │───────▶│  Matching Head    │       │
│                  │  Encoder  │        │  (R³ × SO(3))    │       │
│                  └──────────┘        └────────┬─────────┘       │
│                                               │                 │
│                                               ▼                 │
│                                      ┌──────────────────┐       │
│                                      │  Conformal        │       │
│                                      │  Safety Filter    │       │
│                                      └──────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Motivation

| Approach | Example | What It Optimizes | Limitation for Robotics |
|---|---|---|---|
| VLM-based VLA | OpenVLA, RoboVLM | Image-text alignment | No temporal modeling, semantic bias |
| Video generation | UniPi, SuSIE | Pixel-level prediction | Expensive planning, irrelevant detail |
| **JEPA (ours)** | **VL-JEPA** | **Representation-space dynamics** | **None (this is the repo)** |

V-JEPA 2's key insight (from [LeCun 2022](https://openreview.net/forum?id=BZ5a1r-kVsf)): predict *what changes* in representation space, not pixels. This naturally learns contact dynamics, object motion, and tool use without explicit supervision.

## Target Benchmarks

These are **aspirational targets** based on related work, not reported results:

| Benchmark | Metric | Current SOTA | Target | Source |
|---|---|---|---|---|
| LIBERO Spatial | Success % | ~97-98% | >96% | [LIBERO paper](https://arxiv.org/abs/2306.03310) |
| LIBERO Object | Success % | ~95-97% | >95% | |
| LIBERO Goal | Success % | ~95-97% | >95% | |
| LIBERO Long | Success % | ~90-95% | >92% | |
| CALVIN ABC→D | Avg Length | 4.44 (DreamVLA) | >4.3 | [DreamVLA](https://arxiv.org/abs/2507.04447) |
| MetaWorld MT10 | Success % | ~85-89% | >85% | Various |

**Note**: LIBERO is [essentially saturated](https://mbreuss.github.io/blog_post_iclr_26_vla.html) — properly tuned Diffusion Policies hit 95%+ without VLMs. The real test is CALVIN and real-world transfer.

## Quick Start

```bash
git clone https://github.com/lexus-x/jepa.git
cd jepa

conda create -n vljepa python=3.10 -y
conda activate vljepa
pip install -e ".[dev]"

# Download V-JEPA 2 checkpoint (from Meta)
bash scripts/download_vjepa2.sh

# Verify installation
python tests/smoke_test.py
```

See [docs/setup.md](docs/setup.md) for full installation (including LIBERO's MuJoCo requirements).

## Repository Structure

```
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── docs/
│   ├── setup.md              # Installation & dependencies
│   ├── training.md           # 4-phase training pipeline
│   ├── evaluation.md         # Benchmark evaluation
│   ├── architecture.md       # Design rationale & ablations
│   ├── ssh_workflow.md       # Remote dev workflow
│   └── reproducibility.md    # Seeds, determinism, tracking
├── configs/                  # YAML training configs
├── scripts/                  # Entry points
├── vljepa/                   # Core library
│   ├── models/
│   │   ├── backbone.py       # V-JEPA 2 encoder wrapper
│   │   ├── language.py       # T5 language encoder
│   │   ├── flow_matching.py  # SE(3) flow matching head
│   │   ├── neural_ode.py     # Adaptive Neural ODE
│   │   └── conformal.py      # Conformal prediction
│   ├── data/                 # Datasets & transforms
│   ├── training/             # Trainer, losses, schedulers
│   └── evaluation/           # LIBERO/MetaWorld eval harness
└── tests/
```

## Training Pipeline

| Phase | What | Data | Time (4×A100) |
|---|---|---|---|
| 1 | V-JEPA 2 pretrained | 1M+ hours video (Meta's checkpoint) | — |
| 2 | Action-conditioned post-training | 62h DROID dataset | ~48h |
| 3 | Conformal calibration | Calibration split | ~2h |
| 4 | Benchmark fine-tuning | LIBERO / MetaWorld | ~12h each |

See [docs/training.md](docs/training.md) for exact commands.

## Citation

```bibtex
@article{vjepa2,
  title   = {V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author  = {Assran, Mido and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and others},
  journal = {arXiv preprint arXiv:2506.09985},
  year    = {2025}
}
```

## License

Apache 2.0. V-JEPA 2 checkpoints are subject to [Meta's model license](https://github.com/facebookresearch/vjepa2).
