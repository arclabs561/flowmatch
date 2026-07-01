# flowmatch examples

## Where to start

| I want to... | Run |
|---|---|
| See the smallest semidiscrete flow matching loop | `sd_fm_semidiscrete_linear` |
| Compare semidiscrete FM with minibatch OT pairing | `rfm_minibatch_ot_linear` |
| Check ODE solver error on a known trajectory | `ode_comparison` |
| Train on a 2D multimodal target | `rfm_two_moons` |
| Add class conditioning | `rfm_conditional_2d` |
| Use Fenchel-Young regularizers as Bregman energies | `fynch_bregman_energy` |
| Inspect the low-level OT-CFM training step | `ot_cfm_training_loop` |
| Evaluate samples with MMD | `mmd_flow_eval` |
| See discrete CTMC path evolution | `discrete_ctmc_path_evolution` |

```sh
cargo run --release --example sd_fm_semidiscrete_linear
cargo run --release --example rfm_minibatch_ot_linear
cargo run --release --example ode_comparison
cargo run --release --example rfm_two_moons
cargo run --release --example rfm_conditional_2d
cargo run --release --example fynch_bregman_energy
cargo run --release --example ot_cfm_training_loop
cargo run --release --example mmd_flow_eval
cargo run --release --example discrete_ctmc_path_evolution
```

## Real data

These examples use small checked-in data artifacts.

| Data | Examples | What to read from the output |
|---|---|---|
| Protein torsions | `rfm_protein_torsions_1bpi`, `rfm_torsions_nfe_curve`, `profile_breakdown_torsions` | Ramachandran JS divergence, NFE curve, training-time split |
| USGS earthquakes | `rfm_usgs_earthquakes_sphere`, `rfm_usgs_nfe_curve`, `rfm_usgs_solver_nfe_tradeoff`, `profile_breakdown_usgs` | OT cost, Euler vs Heun at equal NFE, timing split |
| Text-like tokens | `rfm_textish_tokens` | Weighted support matching over token embeddings |

```sh
cargo run --release --example rfm_protein_torsions_1bpi
cargo run --release --example rfm_torsions_nfe_curve
cargo run --release --example rfm_usgs_earthquakes_sphere
cargo run --release --example rfm_usgs_nfe_curve
cargo run --release --example rfm_usgs_solver_nfe_tradeoff
cargo run --release --example rfm_textish_tokens
```

## Failure modes and diagnostics

| Example | What it checks |
|---|---|
| `rfm_minibatch_outlier_partial` | Full minibatch OT can force a source point onto a rare outlier; partial pairing reduces that displacement |
| `profile_breakdown_usgs` | Time spent in sampling, pairing, SGD, and evaluation |
| `profile_breakdown_torsions` | Same timing split on torsion data |

```sh
cargo run --release --example rfm_minibatch_outlier_partial
cargo run --release --example profile_breakdown_usgs
cargo run --release --example profile_breakdown_torsions
```

## Feature-gated examples

| Feature | Examples |
|---|---|
| `burn` | `burn_sd_fm_semidiscrete_linear`, `burn_rfm_minibatch_ot_linear` |
| `riemannian` | `rfm_poincare_geodesic_ode`, `riemannian_fm_poincare` |
| `sheaf-evals` | `rfm_usgs_earthquakes_cluster_mass`, `rfm_usgs_knn_leiden`, `rfm_usgs_full_pipeline_report` |

```sh
cargo run --release --features burn --example burn_sd_fm_semidiscrete_linear
cargo run --release --features burn --example burn_rfm_minibatch_ot_linear
cargo run --release --features riemannian --example rfm_poincare_geodesic_ode
cargo run --release --features riemannian --example riemannian_fm_poincare
cargo run --release --features sheaf-evals --example rfm_usgs_earthquakes_cluster_mass
cargo run --release --features sheaf-evals --example rfm_usgs_knn_leiden
cargo run --release --features sheaf-evals --example rfm_usgs_full_pipeline_report
```

`rfm_usgs_full_pipeline_report` also writes a structured JSON report when
`FLOWMATCH_REPORT_OUT` is set:

```sh
FLOWMATCH_REPORT_OUT=target/flowmatch-usgs-report.json \
  cargo run --release --features sheaf-evals --example rfm_usgs_full_pipeline_report
```

## Shared data loaders

`examples/common/` holds the checked-in torsion, USGS, and text-token helpers used by the examples above. They are example-only utilities, not public API.
