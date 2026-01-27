# flowmatch roadmap: RFM, discrete FM, and ecosystem boundaries

This document is a design/architecture note for evolving `flowmatch` beyond the current
“semidiscrete conditional FM + linear path + Euler sampling” baseline.

## What `flowmatch` is today (and what it is not)

Current implementation (in code):

- **Conditional FM (CFM)** with a **linear path** \(x_t = (1-t)x_0 + t y_j\) and target \(u_t = y_j - x_0\).
- Conditioning variable is **discrete**: \(z := j\), where \(j\) is chosen by a semidiscrete OT-shaped assignment
  implemented in `wass::semidiscrete` (potentials + hard assignment).
- Vector field parameterization is intentionally minimal (`linear::LinearCondField`) and training is simple SGD.
- Sampling integrates an ODE with explicit Euler (and now also Heun/RK2 via `ode::OdeMethod`).

Non-goals (today):

- No SE(3)-equivariant geometry, no manifold-aware flows, no CTMC discrete-time generators.
- No “full reproduction” of any paper; the crate is a small reusable primitive.

## Paper taxonomy → module boundaries (what we should mean by names)

The survey (arXiv:2507.17731) uses a taxonomy that is useful for naming modules:

- **General FM / Conditional FM (CFM)**: regress \(u_\theta(x,t)\) to a tractable conditional target \(u^\*(x,t,z)\).
- **Rectified FM (RFM)**: choose/improve the coupling \(\pi(x_0, x_1)\) to “straighten” flows (reduce curliness).
- **Non-Euclidean FM**: replace Euclidean paths with geodesics / manifold tangent vector fields.
- **Discrete FM**:
  - **CTMC-based**: learn a continuous-time Markov generator over discrete states.
  - **Simplex-based**: learn a continuous flow on the probability simplex (Dirichlet / Fisher-Rao / Gumbel-softmax).

In our crate, those should correspond to *minimal building blocks*, not full application models.

### What is now implemented (scaffolding + baselines)

- **RFM coupling + training baseline**:
  - coupling primitive: `rfm::minibatch_ot_greedy_pairing` (Sinkhorn plan → greedy matching)
  - training: `sd_fm::train_rfm_minibatch_ot_linear` (minibatch OT pairing → straight-line FM regression)
- **Discrete FM scaffolding**:
  - CTMC: `discrete_ctmc` (generator validation + Euler evolution of probabilities)
  - Simplex: `simplex` (simplex validation + Dirichlet sampling)
- **Non-Euclidean scaffolding**: `non_euclidean` (geodesic interpolant trait + Euclidean baseline)

## RFM in our setting (what changes, what stays)

### What stays the same

- The FM loss stays a **velocity regression** on samples \((t, x_t)\).
- We still need a callable vector field \(v_\theta(x,t;\text{cond})\) and an ODE sampler.

### What changes (the key design choice)

RFM is mostly about the **coupling** (how we pair a base sample with a target sample).

In our current SD-FM baseline, the “pairing” is:

- draw \(x_0 \sim \mathcal{N}(0,I)\)
- pick discrete \(j\) by a semidiscrete assignment (OT-flavored)
- set \(x_1 := y_j\)

This is already “rectification-ish” in spirit: the semidiscrete assignment is a coupling between a continuous base
and a discrete support. The *next* step (to deserve the name `rfm`) is to make the coupling choice explicit and swappable.

### Proposed API surface (design)

Introduce a coupling abstraction:

- `CouplingSampler`: given a minibatch of base samples and a minibatch of target samples, return a coupling artifact:
  - simplest: a permutation / matching (one-to-one)
  - more general: a sparse transport plan / weights

Then define “rectified training” as: sample \((x_0, x_1) \sim \pi\) and train on the straight-line path between them.

Where to get \(\pi\) in our ecosystem:

- Prefer using `wass` for approximate OT couplings (entropic OT / Sinkhorn-style), since that’s already our coupling primitive.

What not to do:

- Don’t silently “pretend” a coupling is OT if it isn’t; call it `*_heuristic_*` if it’s a heuristic.

## Discrete FM: two different things (don’t conflate)

### A) CTMC discrete FM (discrete states; rates)

This is for token/graph-like discrete states where the dynamics are a CTMC:

- You model a time-dependent generator \(Q_\theta(t)\) (rates between states).
- Training typically matches probability flows / pathwise objectives, not Euclidean velocities.

**Recommendation**: keep CTMC discrete FM *out of* the current `flowmatch` crate unless we commit to a clean discrete core.
It’s a different math object than “vector field on \(\mathbb{R}^d\)”.

If/when added, it should live in `flowmatch::discrete_ctmc` (or a separate crate) with:

- explicit state representation
- explicit normalization / validity constraints
- deterministic RNG seeds for any sampling

### B) Simplex-based discrete FM (continuous flow on the simplex)

Here you represent categorical data via a point in the simplex and learn a continuous flow there
(Dirichlet FM, Fisher-FM, Gumbel-softmax interpolants).

This is closer to our current continuous-time machinery, but it requires:

- simplex constraints (nonnegativity, sum-to-1)
- stable parameterizations (e.g. logits + softmax with temperature)

**Recommendation**: treat this as a separate module family `flowmatch::simplex_*`
with small, carefully documented invariants (no silent renormalization).

## Ecosystem boundary: backend-agnostic “core” vs ndarray implementation

Right now `flowmatch` exports `ndarray::Array*` types in public structs (`TrainedSdFm`), which means the crate is
not truly backend-agnostic at the public API boundary.

If we want to follow the workspace-wide “backend-agnostic by default” rule, the direction is:

- `flowmatch-core`:
  - traits and scalar math (slice-based inputs/outputs)
  - concepts: interpolants, objectives, coupling interfaces, ODE stepping traits
  - **no `ndarray` in public types**
- `flowmatch-ndarray` (or keep name `flowmatch` and add feature-gated backend modules):
  - the current implementation using `ndarray`
  - test/demo harnesses
  - optional adapters

Current status (as of the latest edits in this workspace): we now have a **feature-gated** Burn-backed
foothold in `flowmatch::burn_euclidean` (compile-tested with `cargo test -p flowmatch --features burn`),
without changing the default ndarray-only API surface.

This split is the clean way to make FM utilities reusable by other crates without importing `ndarray`.

## Suggested next increments (small, testable)

### 1) Make “coupling” explicit in SD-FM training

Refactor `train_sd_fm_semidiscrete_linear` so “choose j” is an injected strategy (still defaulting to the current method).
Add a tiny test that swapping the coupling strategy changes assignment frequencies in the expected direction.

### 2) Add a second ODE integrator

Add Heun/RK2 alongside Euler for sampling, behind a tiny trait. This is low-risk and improves stability.

### 3) Add an RFM-style minibatch coupling utility (optional)

If `wass` exposes a small OT coupling for two minibatches, add:

- `rfm::minibatch_ot_coupling(...) -> matching`
- an e2e test that RFM coupling reduces “path curvature proxy” (e.g. fewer solver steps needed for same MSE).

### 4) Discrete FM prototype (only if we commit)

Pick exactly one:

- `simplex_dirichlet` (continuous-time, simplex constraints), or
- `discrete_ctmc` (true CTMC generator)

and build one minimal invariant test (normalization / probability preservation / determinism).

## References (for naming + scope)

- Lipman et al., *Flow Matching for Generative Modeling* (arXiv:2210.02747)
- Lipman et al., *Flow Matching Guide and Code* (arXiv:2412.06264)
- Li et al., *Flow Matching Meets Biology and Life Science: A Survey* (arXiv:2507.17731)
  - curated list: `https://github.com/Violet24K/Awesome-Flow-Matching-Meets-Biology`

