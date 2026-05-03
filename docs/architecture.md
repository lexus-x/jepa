# VL-JEPA Architecture Notes

## Design Philosophy

VL-JEPA is built on a single thesis: **better mathematical modeling beats more parameters**.

While existing VLAs scale to 7B+ parameters with brute-force VLM backbones, VL-JEPA achieves
superior performance with <500M parameters by respecting the geometric structure of robot actions.

## Why V-JEPA 2 (Dynamics > Semantics)

| Feature | VLMs (PaliGemma, LLaVA) | V-JEPA 2 |
|---------|------------------------|----------|
| Pretraining | Language-image alignment | Video dynamics prediction |
| Learns | "What objects are" | "What will happen next" |
| Output | CLS token or sparse tokens | Dense per-patch features (256/frame) |
| Action relevance | Indirect (semantic → action) | Direct (dynamics → action) |

V-JEPA 2 was pretrained on 1M+ hours of internet video to predict masked spatiotemporal regions
in latent space. This produces representations that inherently encode:
- Object trajectories (what's moving, how fast, in what direction)
- Causal structure (pushing here causes that to move)
- Temporal dynamics (what happens next)

For action generation, dynamics representations are strictly more useful than semantic representations.

## Why SE(3) Flow Matching (Geometry Matters)

Robot end-effector poses live on the SE(3) manifold, not in flat Euclidean space. Predicting
actions in R⁷ introduces rotational artifacts (e.g., averaging two valid rotations can produce
an invalid one).

**SE(3) = SO(3) ⋉ ℝ³** (semidirect product of rotations and translations)

VL-JEPA generates actions as geodesics on SE(3) via Riemannian flow matching:
- **Training**: Conditional Flow Matching with geodesic interpolation
  `γ(t) = exp(t · log(T₁T₀⁻¹))T₀`
- **Inference**: ODE integration from noise to action on the manifold
- **Benefit**: No rotational artifacts, physically plausible trajectories

The task-conditioned metric tensor g_ij(z) learns task-specific geometry:
- Insertion tasks: prioritize translational precision near contact
- Free-space motion: geodesic is nearly straight (low curvature)
- Obstacle avoidance: geodesic curves around obstacles

## Why Adaptive Neural ODE (Computation = Complexity)

Existing VLAs use fixed computation per action — same FLOPs for "move straight" and
"insert peg while rotating." This either under-computes hard tasks or wastes compute on easy ones.

VL-JEPA uses a Neural ODE whose step count adapts to task complexity:
- **Easy actions** (free-space motion): 3-5 ODE steps
- **Hard actions** (contact-rich): 15-20 ODE steps
- **Mathematical guarantee**: N = O(κ(M)/ε) where κ is sectional curvature

This is not a design choice — it's a mathematical consequence of solving the ODE on a curved manifold.

## Why Conformal Prediction (Provable Safety)

Existing VLAs have no mechanism to detect when they're failing. VL-JEPA provides
distribution-free safety guarantees via conformal prediction on SE(3):

- **Nonconformity score**: `s(T_pred, T_true) = ‖log(T_pred⁻¹ T_true)‖` (geodesic distance)
- **Prediction set**: `C_α = {T ∈ SE(3) : d_geo(T_pred, T) ≤ q̂_{1-α}}`
- **Guarantee**: `P(T_true ∈ C_α) ≥ 1 - α` (no distributional assumptions)
- **Online adaptation**: Gibbs-Candès update for non-stationary environments
- **Safety halt**: When conformal radius exceeds threshold → stop and request help

## Ablation Study Design

| Ablation | What's removed | Expected drop | Validates |
|----------|---------------|---------------|-----------|
| VL-JEPA (full) | — | — | Full system |
| w/o task-conditioned metric | Fixed bi-invariant SE(3) metric | -3-5% LIBERO | Metric learning |
| w/o adaptive ODE | Fixed 10 steps | -2-3% on hard tasks | Adaptive computation |
| w/o conformal | No safety sets | 0% accuracy, but safety violations | Conformal safety |
| w/o V-JEPA 2 | Use DINOv2/CLIP backbone | -8-12% across all | Video pretraining |
| w/ Euclidean actions | Predict in R⁷, not SE(3) | -4-6% on contact tasks | Geodesic benefit |
| w/ 300M only (no flow head) | Linear probe on V-JEPA 2 | -15-20% | Flow matching necessity |

## Parameter Budget

| Component | Params | Trainable |
|-----------|--------|-----------|
| V-JEPA 2 Encoder (frozen) | 300M | 0 |
| Language Adapter | 8M | 8M |
| Metric Network | 15M | 15M |
| Geodesic Flow Head | 120M | 120M |
| Conformal Module | 30M | 30M |
| **Total** | **473M** | **173M** |

## Expected Latency

| Component | Latency | Control Hz |
|-----------|---------|------------|
| V-JEPA 2 encoder (300M) | 8ms | — |
| Metric network (15M) | 1ms | — |
| Flow matching (5 steps, 120M) | 5ms | — |
| Conformal (30M) | 1ms | — |
| **Total (easy task)** | **15ms** | **67 Hz** |
| Flow matching (15 steps, hard) | 15ms | — |
| **Total (hard task)** | **25ms** | **40 Hz** |
