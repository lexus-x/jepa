# VL-JEPA Architecture Notes

## Design Philosophy

VL-JEPA is built on a single thesis: **better mathematical modeling beats more parameters**.

While existing VLAs scale to 7B+ parameters with brute-force VLM backbones, VL-JEPA aims to
achieve competitive performance with <500M parameters by respecting the geometric structure
of robot actions.

**Status**: This is a research framework. Core claims below are hypotheses backed by
theoretical motivation, not empirical results. See "Open Questions" for what remains untested.

## Why V-JEPA 2 (Dynamics > Semantics)

| Feature | VLMs (PaliGemma, LLaVA) | V-JEPA 2 |
|---------|------------------------|----------|
| Pretraining | Language-image alignment | Video dynamics prediction |
| Learns | "What objects are" | "What will happen next" |
| Output | CLS token or sparse tokens | Dense per-patch features (256/frame) |
| Action relevance | Indirect (semantic → action) | Direct (dynamics → action) |

V-JEPA 2 was pretrained on 1M+ hours of internet video to predict masked spatiotemporal regions
in latent space. This produces representations that encode:
- Object trajectories (what's moving, how fast, in what direction)
- Causal structure (pushing here causes that to move)
- Temporal dynamics (what happens next)

**Hypothesis**: For action generation, dynamics representations are more useful than semantic
representations. This has not been validated end-to-end — see "Open Questions".

## Why SE(3) Flow Matching (Geometry Matters)

Robot end-effector poses live on the SE(3) manifold, not in flat Euclidean space. Predicting
actions in R⁷ introduces rotational artifacts (e.g., averaging two valid rotations can produce
an invalid one).

**SE(3) = SO(3) ⋉ ℝ³** (semidirect product of rotations and translations)

VL-JEPA generates actions as geodesics on SE(3) via Riemannian flow matching:
- **Training**: Conditional Flow Matching with geodesic interpolation
  `γ(t) = exp(t · log(T₁T₀⁻¹))T₀`
- **Inference**: Adaptive ODE integration from noise to action on the manifold
- **Benefit**: No rotational artifacts, physically plausible trajectories

### Task-Conditioned Metric Tensor

The metric tensor g_ij(z) is **learned and conditioned on the task embedding** (fused visual +
language features), not a fixed diagonal:

```
g(z) = diag(softplus(Wz + b))  ∈ ℝ^{6×6}
```

This lets the loss landscape adapt per-task:
- Insertion tasks: upweight translational precision near contact
- Free-space motion: uniform metric (geodesic is nearly straight)
- Rotation-heavy tasks: upweight rotational components

## Why Adaptive Neural ODE (Computation = Complexity)

Existing VLAs use fixed computation per action — same FLOPs for "move straight" and
"insert peg while rotating." This either under-computes hard tasks or wastes compute on easy ones.

VL-JEPA uses an adaptive ODE integrator (Dormand-Prince RK45) with:

1. **Embedded error estimation**: 5th/4th order pair gives local error estimate per step
2. **PI controller**: adjusts dt based on error ratio, clamped to [dt_min, dt_max]
3. **Curvature-based halting**: stops when velocity norm and curvature drop below threshold
4. **Learned halting** (ponder-net style): auxiliary head predicts halting probability per step

This is not just a fixed step count with a fancy name — it's actual adaptive integration.

**Runtime behavior** (expected, not measured):
- Easy actions (free-space motion): 5-10 ODE steps
- Hard actions (contact-rich): 20-50 ODE steps
- Mathematical motivation: N = O(κ(M)/ε) where κ is sectional curvature

## Why Conformal Prediction (Provable Safety)

Existing VLAs have no mechanism to detect when they're failing. VL-JEPA provides
distribution-free safety guarantees via conformal prediction on SE(3):

- **Nonconformity score**: `s(T_pred, T_true) = ‖log(T_pred⁻¹ T_true)‖` (geodesic distance)
- **Prediction set**: `C_α = {T ∈ SE(3) : d_geo(T_pred, T) ≤ q̂_{1-α}}`
- **Guarantee**: `P(T_true ∈ C_α) ≥ 1 - α` (no distributional assumptions)
- **Online adaptation**: Gibbs-Candès update for non-stationary environments
- **Safety halt**: When conformal radius exceeds threshold → fallback to conservative action

### Deployment Integration

The `SafePolicyWrapper` wires conformal prediction into the actual control loop:

```
action = safe_policy.act(images, instruction, proprio, encoder)
# Internally: policy predicts → calibrator checks radius → fallback if unsafe
```

When ground truth is available, the calibrator updates online:
```
info = safe_policy.update(predicted_action, true_action)
# Returns: scores, coverage, radius, whether fallback was needed
```

## Open Questions (Untested Claims)

The following are **hypotheses** that require experimental validation. They are not
claimed as facts.

### 1. JEPA vs VLM for action prediction

**Claim**: V-JEPA 2 features outperform VLM features (CLIP, DINOv2, PaliGemma) for SE(3) action
generation.

**What's needed**:
- Controlled experiment: same flow matching head, same training data, different encoders
- Benchmarks: LIBERO (all 4 suites), MetaWorld MT10/MT50
- Metrics: success rate, geodesic error, sample efficiency

**Status**: Not run. The codebase has the infrastructure to run this comparison, but no
results exist.

### 2. Task-conditioned metric vs fixed metric

**Claim**: A learned, task-conditioned metric tensor outperforms a fixed bi-invariant metric.

**What's needed**:
- Ablation: fixed [1,1,1,1,1,1] vs learned g(z)
- Measure: loss convergence speed, final success rate per task type

**Status**: Not run.

### 3. Adaptive ODE vs fixed-step integration

**Claim**: Adaptive-step integration outperforms fixed-step (Euler/RK4) at equal or less compute.

**What's needed**:
- Ablation: adaptive vs Euler-10 vs Euler-20 vs RK4-10
- Measure: success rate vs total FLOPs, latency distribution

**Status**: Not run.

### 4. Conformal safety in closed-loop control

**Claim**: Conformal prediction sets provide valid coverage in closed-loop robotic control.

**What's needed**:
- Coverage test: does the empirical coverage match 1-α?
- Safety test: does the fallback mechanism prevent dangerous actions?
- Distribution shift test: does the Gibbs-Candès adaptation handle domain shift?

**Status**: Not run.

## Ablation Study Design (Pending Experiments)

The following ablations should be run to validate the architecture. **No results exist yet.**

| Ablation | What's removed | Hypothesis | Status |
|----------|---------------|------------|--------|
| VL-JEPA (full) | — | — | Pending |
| w/o task-conditioned metric | Fixed bi-invariant SE(3) metric | Harder tasks benefit from adaptive metric | Not run |
| w/o adaptive ODE | Fixed 10 steps (Euler) | Adaptive uses less compute on easy tasks | Not run |
| w/o conformal | No safety sets | Same accuracy, but no safety guarantees | Not run |
| VLM encoder (ablation) | Replace V-JEPA 2 with DINOv2/CLIP | V-JEPA dynamics features help action prediction | Not run |
| w/ Euclidean actions | Predict in R⁷, not SE(3) | SE(3) helps on contact-rich tasks | Not run |
| w/ 300M only (no flow head) | Linear probe on V-JEPA 2 | Flow matching is necessary | Not run |

**To run these ablations**, use:
```bash
# Full model
./scripts/train.sh --config=configs/config.yaml

# Ablation: fixed metric
./scripts/train.sh --config=configs/config.yaml \
    model.flow_matching.task_dim=0

# Ablation: fixed-step ODE
./scripts/eval.sh --checkpoint=checkpoints/best.pt \
    model.inference.method=euler model.inference.num_steps=10
```

## Parameter Budget (Approximate)

| Component | Params | Trainable | Notes |
|-----------|--------|-----------|-------|
| V-JEPA 2 Encoder (frozen) | 300M | 0 | Unfrozen last 2 layers during fine-tuning |
| Language Adapter (all-mpnet-base-v2) | 109M (frozen) + 12M (learned) | 12M | Cross-attention + spatial reasoning |
| Task-Conditioned Metric | ~500K | 500K | Small MLP: task_dim → 6 |
| Velocity Field (flow head) | ~15M | 15M | 6-layer MLP with FiLM |
| Learned Halting Network | ~10K | 10K | Tiny MLP for halting probability |
| Conformal (LieScorer + calibrator) | ~0 | 0 | Pure computation, no learned params |
| **Total** | **~436M** | **~28M** | Mostly frozen encoder |

**Note**: The conformal module has zero learned parameters — it's a statistical procedure,
not a neural network. The numbers above are approximate and depend on exact model choices.

## Expected Latency (Theoretical, Not Measured)

These are **estimates** based on parameter counts and typical GPU throughput. They have
not been benchmarked.

| Component | Est. Latency | Notes |
|-----------|-------------|-------|
| V-JEPA 2 encoder (300M) | ~8ms | ViT-L/16 on A100 |
| Language encoding | ~2ms | Frozen backbone |
| Flow matching (adaptive, easy) | ~5ms | 5-10 ODE steps |
| Flow matching (adaptive, hard) | ~15ms | 20-50 ODE steps |
| Conformal check | <1ms | No neural net |
| **Total (easy task)** | **~16ms** | ~60 Hz |
| **Total (hard task)** | **~26ms** | ~38 Hz |
