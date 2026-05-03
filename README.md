# SE(3) Conformal Prediction for Safe Robot Policies

**Geometric safety guarantees for learned robot policies via conformal prediction on SE(3).**

## The Problem

VLAs fail silently.  When OpenVLA, π₀, or JEPA-VLA makes a mistake, there's no
mechanism to detect it or provide a safe fallback.  Existing failure detectors
(SAFE) reduce this to a scalar — you know *something* is wrong, but not *what*
or *where*.

## The Approach

We wrap **any VLA** with conformal prediction on SE(3), producing geodesic ball
prediction sets with **distribution-free coverage guarantees**:

```python
from src.policies import OpenVLAPolicy
from src.conformal import SafePolicyWrapper, OnlineConformalCalibrator

# Wrap any VLA with conformal safety
policy = OpenVLAPolicy(model_name="openvla/openvla-7b")
safe = SafePolicyWrapper(
    policy=policy,
    calibrator=OnlineConformalCalibrator(alpha=0.1),  # 90% coverage
    max_radius=2.0,  # fallback if radius exceeds this
)

# Use like a normal policy
action, info = safe.act(observation, "pick up the red cup")
# action: [1, 4, 4] SE(3) pose
# info: {"radius": 0.3, "fallback": False, "halted": False}
```

## Why SE(3) Conformal > Scalar Detectors

| | SAFE (scalar) | SE(3) Conformal |
|--|--------------|-----------------|
| Output | P(failure) ∈ [0,1] | Ball in SE(3) with radius r |
| Geometry | None | Rotation vs translation breakdown |
| Guarantee | Heuristic | P(T_true ∈ C_α) ≥ 1-α |
| Adaptation | Fixed threshold | Gibbs-Candès online |
| Failure mode | "Something's wrong" | "Rotation is off by 0.3 rad" |

## Quick Start

```bash
# Random policy baseline (no VLA needed, tests the conformal layer)
./scripts/eval.sh --policy=random --all

# With a real VLA
./scripts/eval.sh --policy=openvla --suite=libero_spatial

# Custom conformal parameters
./scripts/eval.sh --policy=pi0 --alpha=0.05 --max_radius=1.5
```

## Supported Policies

| Policy | Adapter | Notes |
|--------|---------|-------|
| OpenVLA | `OpenVLAPolicy` | 7D → SE(3) conversion |
| π₀ | `PiZeroPolicy` | Action chunk → first action → SE(3) |
| JEPA-VLA | `JEPAPolicy` | Native SE(3) output |
| Random | `RandomPolicy` | Baseline for testing |

## How It Works

1. **LieScorer** computes nonconformity: `s(T_pred, T_true) = ‖log(T_pred⁻¹ T_true)‖`
2. **OnlineConformalCalibrator** tracks scores with Gibbs-Candès adaptation
3. **SafePolicyWrapper** checks radius before returning each action
4. If radius > threshold → fallback (identity = no movement)
5. Statistics tracked: coverage rate, intervention rate, radius evolution

## Key Hypotheses

1. **SE(3) conformal sets are tighter** than scalar detectors at matched coverage
2. **Adaptive ODE steps correlate with conformal radius** (more steps = less certain)
3. **Gibbs-Candès handles distribution shift** (valid coverage under domain change)

See [docs/architecture.md](docs/architecture.md) for the full experimental design.

## Project Structure

```
src/
├── policies/           # VLA adapters (any VLA → SE(3))
├── conformal/          # Core: LieScorer, OnlineCalibrator, SafeWrapper
├── flow/se3_*.py       # SE(3) Lie group utilities
└── evaluation/         # LIBERO, MetaWorld, SAFE comparison
```

## License

MIT
