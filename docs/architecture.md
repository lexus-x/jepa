# SE(3) Conformal Prediction for Safe Robot Policies

## Problem

Learned robot policies (VLAs) fail silently.  When a policy makes a mistake,
there's no built-in mechanism to detect the failure or provide a safe fallback.
Existing approaches (SAFE, confidence thresholds) reduce failure detection to
a **scalar** — a single number that tells you *something* is wrong, but not
*what* or *where*.

## Approach

We wrap any VLA with **conformal prediction on SE(3)**, the manifold of rigid
body transformations.  Instead of a scalar failure score, we produce a
**geodesic ball prediction set**:

```
C_α = {T ∈ SE(3) : d_geo(T_pred, T) ≤ q̂_{1-α}}
```

This provides:

| Property | SAFE (scalar) | SE(3) Conformal |
|----------|--------------|-----------------|
| Output | P(failure) ∈ [0,1] | Ball in SE(3) with radius r |
| Geometry | None | Rotation vs translation breakdown |
| Guarantee | Heuristic | P(T_true ∈ C_α) ≥ 1-α (distribution-free) |
| Adaptation | Fixed threshold | Gibbs-Candès online update |
| Set shape | Scalar threshold | Geodesic ball (task-conditioned) |

## Architecture

```
┌─────────────┐
│  Any VLA    │  OpenVLA, π₀, JEPA-VLA, ...
│  (frozen)   │
└──────┬──────┘
       │ T_pred ∈ SE(3)
       ▼
┌──────────────────────────────────────────┐
│         SafePolicyWrapper                │
│                                          │
│  ┌─────────────┐  ┌───────────────────┐  │
│  │ LieScorer   │  │ OnlineConformal   │  │
│  │             │──│ Calibrator        │  │
│  │ s(T,T') =   │  │                   │  │
│  │ ‖log(T⁻¹T')‖│  │ α_{t+1} = α_t +  │  │
│  │             │  │ η(α - 𝟙[missed])  │  │
│  └─────────────┘  └───────────────────┘  │
│          │                │               │
│          ▼                ▼               │
│     ┌─────────────────────────┐          │
│     │  if r > r_max → fallback│          │
│     │  if halted   → fallback │          │
│     │  else        → T_pred   │          │
│     └─────────────────────────┘          │
└──────────────────────────────────────────┘
```

## Components

### LieScorer (`src/conformal/lie_scorer.py`)

Computes nonconformity scores on SE(3):

```
s(T_pred, T_true) = ‖log(T_pred⁻¹ T_true)‖_g
```

where g is a weighted norm (configurable rotation/translation weights).
Also provides:
- Geodesic ball prediction sets
- Per-axis error breakdown (rotation vs translation)
- Calibration on held-out trajectories

### OnlineConformalCalibrator (`src/conformal/online_calibration.py`)

Gibbs-Candès adaptive conformal prediction:

```
α_{t+1} = α_t + η(α - 𝟙[s_t > q̂_t])
```

Features:
- Exponentially decaying weights for old scores (handles drift)
- Streaming quantile estimation (no full history needed)
- Safety halt when radius exceeds threshold
- Serializable state for checkpointing

### SafePolicyWrapper (`src/conformal/safe_policy.py`)

Wraps any `BasePolicy` with conformal safety:
1. Calls `policy.predict_action(observation, instruction)`
2. Checks conformal radius against threshold
3. Falls back to identity (no movement) if unsafe
4. Tracks coverage, radius, and intervention statistics

### Policy Adapters (`src/policies/`)

Drop-in adapters for existing VLAs:
- `OpenVLAPolicy`: Wraps OpenVLA (7D → SE(3))
- `PiZeroPolicy`: Wraps π₀ (action chunk → SE(3))
- `JEPAPolicy`: Wraps JEPA-VLA (native SE(3))
- `RandomPolicy`: Baseline for testing

## Key Hypotheses (To Be Tested)

### H1: SE(3) conformal sets are tighter than scalar detectors

**Claim**: For the same coverage level (1-α), SE(3) geodesic ball prediction
sets are smaller (in volume) than the equivalent scalar threshold region.

**Why this might be true**: The geodesic ball respects the manifold structure.
A scalar detector in R⁷ wastes volume on rotations that are physically
impossible (e.g., the "average" of two valid rotations may be invalid).

**Experiment**: Compare `conformal_set_size` (r⁶) vs `safe_set_size` (scalar
threshold) at matched coverage rates.

### H2: Adaptive ODE steps correlate with conformal radius

**Claim**: When the adaptive ODE integrator takes more steps (higher task
complexity), the conformal radius is larger (more uncertainty).

**Why this might be true**: More ODE steps = higher curvature = the flow
is harder to integrate = the model is less certain = larger prediction set.

**Experiment**: Correlate ODE step count with conformal radius across tasks.

### H3: Gibbs-Candès adaptation handles distribution shift

**Claim**: The online calibrator maintains valid coverage when the test
distribution differs from calibration (e.g., new objects, new scenes).

**Why this might be true**: Gibbs-Candès updates α_t based on recent
coverage feedback, so it adapts to shifting distributions.

**Experiment**: Evaluate on out-of-distribution tasks, measure coverage
over time.

## Benchmark Protocol

### Metrics

1. **Coverage rate**: Fraction of true actions inside C_α (target: ≥ 1-α)
2. **Conformal radius**: Mean and std of prediction set radius
3. **Safety intervention rate**: Fraction of steps using fallback
4. **Localization**: Rotation vs translation error breakdown
5. **Adaptation speed**: Steps to reach target coverage after shift

### Baselines

1. **SAFE** (scalar failure detector): Train MLP on se(3) features
2. **No safety**: Raw VLA without any safety wrapper
3. **Fixed threshold**: Conformal with fixed (non-adaptive) calibration

### Benchmarks

1. **LIBERO**: 4 suites (Spatial, Object, Goal, Long) × 10 tasks
2. **MetaWorld**: MT10, MT50
3. **Distribution shift**: Train on LIBERO-Spatial, test on LIBERO-Long

## Running

```bash
# Random policy baseline (no VLA needed)
./scripts/eval.sh --policy=random --all

# OpenVLA with conformal safety
./scripts/eval.sh --policy=openvla --suite=libero_spatial

# Custom alpha and radius
./scripts/eval.sh --policy=pi0 --alpha=0.05 --max_radius=1.5

# Python API
python -m src.run_eval --policy=random --benchmark=libero --suite=libero_spatial
```

## File Structure

```
src/
├── policies/
│   ├── base.py           # Abstract BasePolicy interface
│   ├── openvla.py        # OpenVLA adapter
│   ├── pi0.py            # π₀ adapter
│   ├── jepa_vla.py       # JEPA-VLA adapter
│   └── random.py         # Random baseline
├── conformal/
│   ├── lie_scorer.py     # SE(3) nonconformity scores
│   ├── online_calibration.py  # Gibbs-Candès adaptive conformal
│   └── safe_policy.py    # Safety wrapper for any VLA
├── flow/
│   ├── se3_utils.py      # SE(3) exp/log/geodesic
│   └── se3_manifold.py   # SE3 manifold for flow_matching
└── evaluation/
    ├── libero_eval.py    # LIBERO benchmark
    ├── metaworld_eval.py # MetaWorld benchmark
    └── comparison.py     # SAFE comparison framework
```
