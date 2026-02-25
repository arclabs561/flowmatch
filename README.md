# flowmatch

Flow matching in Rust. Train generative models that learn to transport noise to data via ODE vector fields.

## What it provides

**Training**: Semidiscrete FM (`sd_fm`), rectified flow matching with minibatch OT coupling (`rfm`).

**Sampling**: Fixed-step ODE integrators (Euler, Heun) for Euclidean and Riemannian manifolds.

**Coupling**: Sinkhorn OT pairing, greedy matching, partial/selective pairing (handles outliers), cost-normalized variants.

**Discrete FM**: CTMC generator scaffolding with cosine-squared schedule from Gat et al. 2024 (NeurIPS), conditional probability paths, conditional rate matrices.

**Time schedules**: Uniform, U-shaped, logit-normal (Stable Diffusion 3).

**Evaluation**: JS divergence on histograms, entropic OT cost.

## Dependencies (all ours)

- [`wass`](https://github.com/arclabs561/wass) -- optimal transport (Sinkhorn, coupling)
- [`skel`](https://github.com/arclabs561/skel) -- manifold trait (exp/log/transport)
- [`logp`](https://github.com/arclabs561/logp) -- information theory (JS divergence)
- [`hyperball`](https://github.com/arclabs561/hyp) -- hyperbolic geometry (dev-dependency for Riemannian tests)

## Quick start

```bash
# Semidiscrete FM (simplest)
cargo run --release --example sd_fm_semidiscrete_linear

# Rectified flow matching with minibatch OT
cargo run --release --example rfm_minibatch_ot_linear

# Real data: protein torsion angles (Ramachandran evaluation)
cargo run --release --example rfm_protein_torsions_1bpi

# Real data: USGS earthquakes on the sphere
cargo run --release --example rfm_usgs_earthquakes_sphere
```

## All examples

| Example | What it shows |
|---|---|
| `sd_fm_semidiscrete_linear` | Simplest FM: Gaussian noise to discrete targets |
| `rfm_minibatch_ot_linear` | Minibatch OT coupling for straighter trajectories |
| `rfm_minibatch_outlier_partial` | Outlier forcing problem and partial pairing fix |
| `rfm_protein_torsions_1bpi` | Real protein phi/psi angles on the torus, JS divergence metric |
| `rfm_usgs_earthquakes_sphere` | Real earthquake locations on S^2, OT cost metric |
| `rfm_textish_tokens` | Token embeddings with TF-IDF weights |
| `rfm_torsions_nfe_curve` | Sample quality vs. ODE steps (torsion data) |
| `rfm_usgs_nfe_curve` | Sample quality vs. ODE steps (earthquake data) |
| `rfm_usgs_solver_nfe_tradeoff` | Euler vs. Heun under equal compute budgets |
| `rfm_poincare_geodesic_ode` | Riemannian ODE on Poincare ball (`--features riemannian`) |
| `profile_breakdown_*` | Where training time goes (Sinkhorn vs SGD) |

Tier-evals examples (require `--features tier-evals`):

| Example | What it shows |
|---|---|
| `rfm_usgs_earthquakes_cluster_mass` | Structure-aware scoring via cluster mass |
| `rfm_usgs_knn_leiden` | kNN graph + Leiden community detection |
| `rfm_usgs_full_pipeline_report` | Full pipeline with all metrics and timings |

## Tests

```bash
cargo test -p flowmatch                  # 65 tests
cargo test -p flowmatch --features burn  # + burn backend tests
```

## References

- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (2022)
- Lipman et al., [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) (2024)
- Gat et al., [Discrete Flow Matching](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8a2a3efb0b8e1b8cbd7c69bda6a4d2df-Abstract-Conference.html) (NeurIPS 2024)
- Chen & Lipman, [Riemannian Flow Matching](https://arxiv.org/abs/2302.03660) (2023)

## License

MIT OR Apache-2.0
