# flowmatch

Flow matching as a small, backend-agnostic Rust library primitive.

This crate currently focuses on a **semidiscrete flow matching** setup:

- **Discrete target support**: a finite set of prototypes \(y_j\) with weights \(b_j\).
- **Semidiscrete conditioning / assignment**: uses `wass::semidiscrete` potentials + hard assignment to pick an index \(j\).
- **Flow matching regression**: trains a conditional vector field \(v_\theta(x,t; y_j)\) against a simple linear-path target.

Code entrypoints:

- `flowmatch::sd_fm::train_sd_fm_semidiscrete_linear`
- `flowmatch::sd_fm::TrainedSdFm::{sample,sample_with_x0}`
- `flowmatch::linear::LinearCondField` (intentionally “boring baseline”)

Related primitives implemented in this crate:

- `flowmatch::ode`: fixed-step ODE samplers (`Euler`, `Heun`)
- `flowmatch::rfm`: coupling helpers for rectified / OT-based pairing
- `flowmatch::simplex`: simplex validation + Dirichlet sampling (simplex-based “discrete FM” scaffolding)
- `flowmatch::discrete_ctmc`: CTMC generator validation + a minimal probability evolution step
- `flowmatch::non_euclidean`: geodesic interpolant scaffolding (currently includes only Euclidean baseline)

Related (adjacent meaning of “distribution matching”):
- `decipher/`: symbolic distribution matching for classical text deciphers (letter-frequency scoring, etc.). See `canon/topics/distribution-matching.md`.

### References (why this crate is called `flowmatch`)

These are the conceptual anchors for the objective + design space:

- Lipman et al., *Flow Matching for Generative Modeling* (arXiv:2210.02747).  
  Link: [arXiv](https://arxiv.org/abs/2210.02747)
- Lipman et al., *Flow Matching Guide and Code* (arXiv:2412.06264).  
  Link: [arXiv](https://arxiv.org/abs/2412.06264)

Also useful as an applications-oriented map (especially for discrete / non-Euclidean variants):

- Li et al., *Flow Matching Meets Biology and Life Science: A Survey* (arXiv:2507.17731, 2025).  
  Link: [arXiv](https://arxiv.org/abs/2507.17731)  
  Curated resources: `https://github.com/Violet24K/Awesome-Flow-Matching-Meets-Biology`

Related work you surfaced that’s **not implemented** in this crate (yet), but is a good roadmap:

- Chen & Lipman, *Flow Matching on General Geometries* (arXiv:2302.03660) — Riemannian FM  
  Link: [arXiv](https://arxiv.org/abs/2302.03660)
- Dao et al., *Flow Matching in Latent Space* (arXiv:2307.08698) — latent FM and guidance  
  Link: [arXiv](https://arxiv.org/abs/2307.08698)
- Gat et al., *Discrete Flow Matching* (NeurIPS 2024) — discrete state spaces  
  Link: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8a2a3efb0b8e1b8cbd7c69bda6a4d2df-Abstract-Conference.html)
- Klein et al., *Equivariant Flow Matching* (NeurIPS 2023) — symmetry/equivariance constraints  
  Link: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9ebf53f885e4e4d114d5d28ca13f4988-Abstract-Conference.html)

### Running the demo

```bash
cargo run -p flowmatch --example sd_fm_semidiscrete_linear
```

RFM minibatch OT demo:

```bash
cargo run -p flowmatch --example rfm_minibatch_ot_linear
```

RFM demo on token embeddings + TF-IDF-ish weights:

```bash
cargo run -p flowmatch --example rfm_textish_tokens
```

RFM demo on **real USGS earthquake locations** (sphere-ish geodata):

```bash
cargo run -p flowmatch --example rfm_usgs_earthquakes_sphere
```

RFM demo on **real USGS earthquake locations**, evaluated via **cluster-mass structure** (uses `tier`):

```bash
cargo run -p flowmatch --example rfm_usgs_earthquakes_cluster_mass --features tier-evals
```

Full engine composition demo (flowmatch + tier + jin): kNN graph → Leiden communities:

```bash
cargo run -p flowmatch --example rfm_usgs_knn_leiden --features tier-evals
```

Full pipeline report (all metrics + timings, including deterministic exact-kNN Leiden and optional HNSW-kNN):

```bash
cargo run -p flowmatch --example rfm_usgs_full_pipeline_report --features tier-evals
```

NFE/steps curve (paper-style “few-step” evaluation):

```bash
cargo run -p flowmatch --example rfm_usgs_nfe_curve
```

Solver NFE tradeoff (Euler vs Heun under equal evaluation budgets):

```bash
cargo run -p flowmatch --example rfm_usgs_solver_nfe_tradeoff
```

Protein torsions NFE/steps curve (seed-averaged, Ramachandran JS):

```bash
cargo run -p flowmatch --example rfm_torsions_nfe_curve
```

Minibatch OT outlier forcing + partial pairing mitigation:

```bash
cargo run -p flowmatch --example rfm_minibatch_outlier_partial
```

Controls:
- `FLOWMATCH_PAIRING=partial_rowwise` uses `RfmMinibatchPairing::PartialRowwise`
- `FLOWMATCH_PAIRING=sinkhorn_selective`, uses Sinkhorn then selective matching
- `FLOWMATCH_PAIRING_PARTIAL_KEEP_FRAC=0.8` controls the fraction of rows that are forced one-to-one

Speed knobs for the full pipeline report:

```bash
# Default (highest quality): Sinkhorn pairing every step.

# Faster: reuse Sinkhorn pairing for 4 SGD steps (usually ~4× faster coupling).
FLOWMATCH_PAIRING_EVERY=4 cargo run -p flowmatch --example rfm_usgs_full_pipeline_report

# Fastest: no Sinkhorn at all (row-wise nearest pairing).
FLOWMATCH_PAIRING=rowwise cargo run -p flowmatch --example rfm_usgs_full_pipeline_report

# U-shaped timestep sampling (more weight near t=0 and t=1)
FLOWMATCH_T_SCHEDULE=ushaped cargo run -p flowmatch --example rfm_usgs_full_pipeline_report
```

RFM demo on **real protein φ/ψ torsions** (a torus-shaped domain, scored via Ramachandran JS divergence):

```bash
cargo run -p flowmatch --example rfm_protein_torsions_1bpi
```

Timing breakdown (“poor man's profiling”): where time goes (sampling vs Sinkhorn vs SGD):

```bash
cargo run -p flowmatch --example profile_breakdown_usgs
cargo run -p flowmatch --example profile_breakdown_torsions
```

### Running tests

```bash
cargo test -p flowmatch
```

### Burn backend (opt-in)

`flowmatch` is `ndarray`-only by default, but it now includes an **opt-in** Burn-backed Euclidean FM
module behind the `burn` feature (see `flowmatch::burn_euclidean` / `flowmatch::burn_sd_fm`).

```bash
cargo test -p flowmatch --features burn
```

Run the Burn-backed toy examples:

```bash
cargo run -p flowmatch --example burn_sd_fm_semidiscrete_linear --features burn
cargo run -p flowmatch --example burn_rfm_minibatch_ot_linear --features burn
```

