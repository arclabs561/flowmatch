# flowmatch

[![crates.io](https://img.shields.io/crates/v/flowmatch.svg)](https://crates.io/crates/flowmatch)
[![Documentation](https://docs.rs/flowmatch/badge.svg)](https://docs.rs/flowmatch)
[![CI](https://github.com/arclabs561/flowmatch/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/flowmatch/actions/workflows/ci.yml)

Flow matching in Rust. Train generative models that learn to transport noise to data via ODE vector fields.

## Problem

You have a set of target points -- protein backbone angles, earthquake epicenters, token embeddings -- and want to train a vector field that transforms Gaussian noise into samples from the same distribution. Flow matching [1] does this by regressing a conditional vector field along straight (or geodesic) interpolation paths, then sampling via ODE integration.

This library provides the training loop, OT-based coupling, ODE integration, and evaluation metrics. It works on flat spaces and on Riemannian manifolds.

## Examples

**Transport noise to discrete targets** (simplest case). Semidiscrete FM pairs Gaussian noise with fixed target points via optimal transport, trains a linear conditional field, and integrates an ODE to produce samples:

```bash
cargo run --release --example sd_fm_semidiscrete_linear
```

```text
n=16 d=8
pot_cfg: steps=2000 batch=1024 seed=7
fm_cfg:  steps=800 batch=256 lr=0.008 seed=9 euler_steps=40
sample_mse_to_assigned_y = 0.0367
```

**Straighter trajectories via minibatch OT**. Rectified flow matching [7] uses Sinkhorn coupling within each minibatch so that noise-to-data paths cross less, reducing integration error:

```bash
cargo run --release --example rfm_minibatch_ot_linear
```

```text
sample_mse_to_assigned_y = 0.0684
```

**Protein torsion angles on a torus**. Backbone phi/psi angles live on S^1 x S^1. This example trains on real angles from PDB 1BPI (BPTI), then measures sample quality by JS divergence between generated and observed Ramachandran histograms:

```bash
cargo run --release --example rfm_protein_torsions_1bpi
```

```text
PDB 1BPI φ/ψ (n=56) as a torus via R^4 embedding
Ramachandran histogram JS divergence (lower is better):
- baseline (Gaussian decode): 0.6391
- trained  (RFM+minibatch OT): 0.4105
- ratio trained/baseline: 0.642
```

**Earthquake locations on a sphere**. USGS M6+ earthquake epicenters (2024) mapped to S^2. Evaluation uses entropic OT cost between generated and observed locations:

```bash
cargo run --release --example rfm_usgs_earthquakes_sphere
```

```text
USGS earthquakes (n=50), embedding=R^3 with S^2 projection
OT cost (lower is better):
- baseline (near-noise): 0.6496
- trained  (RFM+minibatch OT): 0.3129
- ratio trained/baseline: 0.482

Some generated samples (lat, lon):
   0: lat=  12.63°, lon= -104.96°
   1: lat=  58.20°, lon=  169.16°
   2: lat= -13.11°, lon= -167.62°
   3: lat= -35.47°, lon=  -79.28°
```

**Geodesics on the Poincare ball**. Riemannian ODE integration on hyperbolic space, using the `skel::Manifold` trait implemented by `hyperball`:

```bash
cargo run --release --example rfm_poincare_geodesic_ode --features riemannian
```

### All examples

| Example | What it shows |
|---|---|
| `sd_fm_semidiscrete_linear` | Gaussian noise to discrete targets via semidiscrete OT |
| `rfm_minibatch_ot_linear` | Minibatch Sinkhorn coupling for straighter trajectories |
| `rfm_minibatch_outlier_partial` | Outlier forcing problem and partial pairing fix |
| `rfm_protein_torsions_1bpi` | Real protein phi/psi angles on the torus, JS divergence metric |
| `rfm_usgs_earthquakes_sphere` | Real earthquake locations on S^2, OT cost metric |
| `rfm_textish_tokens` | Token embeddings with TF-IDF weights |
| `rfm_torsions_nfe_curve` | Sample quality vs. ODE steps (torsion data) |
| `rfm_usgs_nfe_curve` | Sample quality vs. ODE steps (earthquake data) |
| `rfm_usgs_solver_nfe_tradeoff` | Euler vs. Heun under equal compute budgets |
| `ode_comparison` | Euler vs Heun on a 2D circular ODE (radius preservation) |
| `rfm_poincare_geodesic_ode` | Riemannian ODE on Poincare ball (`--features riemannian`) |
| `discrete_ctmc_path_evolution` | CTMC path evolution with time-dependent generators |
| `rfm_conditional_2d` | 2D conditional flow matching visualization |
| `rfm_two_moons` | Two-moons distribution transport |
| `burn_sd_fm_semidiscrete_linear` | Semidiscrete FM with Burn backend (`--features burn`) |
| `burn_rfm_minibatch_ot_linear` | RFM with Burn backend (`--features burn`) |
| `profile_breakdown_*` | Where training time goes (Sinkhorn vs SGD) |

Requires `--features sheaf-evals`:

| Example | What it shows |
|---|---|
| `rfm_usgs_earthquakes_cluster_mass` | Do generated samples preserve cluster structure? |
| `rfm_usgs_knn_leiden` | kNN graph + Leiden community detection on generated data |
| `rfm_usgs_full_pipeline_report` | Full pipeline with all metrics and timings |

## What it provides

**Training**: Semidiscrete FM, rectified flow matching with minibatch OT coupling, time schedules (uniform, U-shaped, logit-normal).

**Sampling**: Fixed-step ODE integrators (Euler, Heun) for Euclidean and Riemannian manifolds.

**Coupling**: Sinkhorn OT pairing, greedy matching, partial/selective pairing for outlier handling.

**Discrete FM**: CTMC generator scaffolding with cosine-squared schedule [3], conditional probability paths, conditional rate matrices.

**Evaluation**: JS divergence on histograms, entropic OT cost.

## Dependencies

- [`wass`](https://github.com/arclabs561/wass) -- optimal transport (Sinkhorn, coupling)
- [`skel`](https://github.com/arclabs561/skel) -- manifold trait (exp/log/transport)
- [`logp`](https://github.com/arclabs561/logp) -- information theory (JS divergence)
- [`hyperball`](https://github.com/arclabs561/hyperball) -- hyperbolic geometry (dev-dependency for Riemannian tests)

## Status

Not yet on crates.io (`publish = false`). MSRV: 1.80.

## Tests

```bash
cargo test -p flowmatch                  # 77 tests
cargo test -p flowmatch --features burn  # + burn backend tests
```

## References

1. Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (2022)
2. Lipman et al., [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) (2024) -- comprehensive tutorial
3. Gat et al., [Discrete Flow Matching](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8a2a3efb0b8e1b8cbd7c69bda6a4d2df-Abstract-Conference.html) (NeurIPS 2024) -- CTMC-based discrete FM
4. Chen & Lipman, [Riemannian Flow Matching on General Geometries](https://arxiv.org/abs/2302.03660) (2023)
5. de Kruiff et al., [Pullback Flow Matching on Data Manifolds](https://arxiv.org/abs/2410.04543) (2024) -- FM on implicit manifolds without closed-form exp/log maps
6. Sherry & Smets, [Flow Matching on Lie Groups](https://arxiv.org/abs/2505.08393) (2025) -- specialization to SO(3) and SE(3)
7. Liu et al., [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) (2022) -- rectified flow

## License

MIT OR Apache-2.0
