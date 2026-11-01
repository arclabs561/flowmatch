# flowmatch

[![crates.io](https://img.shields.io/crates/v/flowmatch.svg)](https://crates.io/crates/flowmatch)
[![Documentation](https://docs.rs/flowmatch/badge.svg)](https://docs.rs/flowmatch)

Train conditional vector fields with flow matching.

## Install

```sh
cargo add flowmatch
```

## Problem

You have a set of target points (protein backbone angles, earthquake epicenters, token embeddings) and want to train a vector field that transforms Gaussian noise into samples from the same distribution. Flow matching [1] does this by regressing a conditional vector field along straight (or geodesic) interpolation paths, then sampling via ODE integration.

`flowmatch` provides small training loops, OT-based coupling, ODE integration,
and evaluation metrics. Riemannian training and integration are experimental
and feature-gated. Experimental CTMC primitives are included by default.

## Examples

**Transport noise to discrete targets** (simplest case). Semidiscrete FM pairs Gaussian noise with fixed target points via optimal transport, trains a linear conditional field, and integrates an ODE to produce samples:

```bash
cargo run --release --example sd_fm_semidiscrete_linear
```

Illustrative fixed-seed output:

```text
n=16 d=8
pot_cfg: steps=2000 batch=1024 seed=7
fm_cfg:  steps=800 batch=256 lr=0.008 seed=9 euler_steps=40
sample_mse_to_assigned_y = 0.0367  (x0 baseline = 0.7818)
```

**Minibatch OT coupling**. Flow matching can pair base and target samples with an OT coupling inside each minibatch (OT-CFM). The `rfm` names in this crate refer to that pairing, not to the reflow procedure of rectified flow [5]. This example uses Sinkhorn pairing inside each minibatch and reports assigned-target MSE after ODE sampling:

```bash
cargo run --release --example rfm_minibatch_ot_linear
```

Illustrative fixed-seed output:

```text
sample_mse_to_assigned_y = 0.0684  (x0 baseline = 1.4397)
```

**Protein torsion angles on a torus**. Backbone phi/psi angles live on S^1 x S^1. This example trains on real angles from PDB 1BPI (BPTI), then measures sample quality by JS divergence between generated and observed Ramachandran histograms:

```bash
cargo run --release --example rfm_protein_torsions_1bpi
```

Illustrative fixed-seed output:

```text
PDB 1BPI phi/psi (n=56) as a torus via R^4 embedding
Ramachandran histogram JS divergence (lower is better):
- baseline (Gaussian decode): 0.6391
- trained  (RFM+minibatch OT): 0.4105
- ratio trained/baseline: 0.642
```

**Earthquake locations on a sphere**. USGS M6+ earthquake epicenters (2024) mapped to S^2. Evaluation uses entropic OT cost between generated and observed locations:

```bash
cargo run --release --example rfm_usgs_earthquakes_sphere
```

Illustrative fixed-seed output:

```text
USGS earthquakes (n=50), embedding=R^3 with S^2 projection
OT cost (lower is better):
- baseline (near-noise): 0.6496
- trained  (RFM+minibatch OT): 0.3129
- ratio trained/baseline: 0.482

Some generated samples (lat, lon):
   0: lat=  12.63, lon= -104.96
   1: lat=  58.20, lon=  169.16
   2: lat= -13.11, lon= -167.62
   3: lat= -35.47, lon=  -79.28
```

**Geodesics on the Poincare ball**. Riemannian ODE integration on hyperbolic space, using the `skel::Manifold` trait implemented by `hyperball`:

```bash
cargo run --release --example rfm_poincare_geodesic_ode --features riemannian
```

See [`examples/README.md`](examples/README.md) for the example map, including discrete CTMC, two-moons, conditional 2D, Burn backend variants (`--features burn`), Riemannian Poincare disk (`--features riemannian`), and cluster/community evaluation (`--features sheaf-evals`).

## What it provides

**Training**: Semidiscrete FM, flow matching with minibatch OT coupling, time schedules (uniform, U-shaped, logit-normal).

**Sampling**: Fixed-step ODE integrators (Euler, Heun) for Euclidean and Riemannian manifolds.

**Coupling**: Sinkhorn OT pairing, greedy matching, partial/selective pairing for outlier handling.

**Discrete FM**: CTMC generator scaffolding with cosine-squared schedule [3], conditional probability paths, conditional rate matrices.

**Evaluation**: JS divergence on histograms and entropic OT cost.

The Riemannian and CTMC paths are small demonstrations, not paper
reproductions or benchmark evidence.

The core linear model remains endpoint-conditioned while sampling: after an
assignment it integrates `v(x, t; y)`. Its assigned-target MSE does not
establish unconditional generation or a general coupling improvement.

## Status

Declared MSRV: Rust 1.80 for the default feature set; it is not CI-validated.
The current locked macOS resolution of the optional `burn` backend requires
Rust 1.88 or newer.

The protein and earthquake examples train and score against small checked-in
supports. They demonstrate the training and evaluation paths; their metrics do
not estimate out-of-sample generative quality, protein conformations, or
earthquake forecasts. Riemannian support is experimental.

## Tests

```bash
cargo test -p flowmatch
cargo test -p flowmatch --features burn
cargo test -p flowmatch --features riemannian
cargo test -p flowmatch --features sheaf-evals
```

## References

1. Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (2022)
2. Lipman et al., [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) (2024). Tutorial and code guide
3. Gat et al., [Discrete Flow Matching](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0d629a734b56a642701bba7bc8bb3ed-Abstract-Conference.html) (NeurIPS 2024). CTMC-based discrete FM
4. Chen & Lipman, [Riemannian Flow Matching on General Geometries](https://arxiv.org/abs/2302.03660) (2023)
5. Liu et al., [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) (2022). Rectified flow

## License

MIT OR Apache-2.0
