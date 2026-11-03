# Changelog

## [Unreleased]

## [0.1.10] - 2026-11-03

### Fixed

- `rfm::minibatch_ot_greedy_pairing` (and `ot_cfm::ot_cfm_coupling` through it)
  no longer returns `Domain("sinkhorn coupling did not converge")` on batches
  with a far outlier. When Sinkhorn does not converge, it retries once on the
  cost matrix divided by its maximum, as torchcfm's `normalize_cost` does.
- Training entry points (ndarray, Burn and Riemannian) now return
  `Error::Domain` or `Error::Shape` before any update for a non-positive or
  nonfinite learning rate, nonfinite logit-normal schedule parameters,
  nonfinite support points or weights, and `sample_x0` output of the wrong
  length or with nonfinite values.
- `ode` and `riemannian_ode` Euler and Heun integration return `Error::Shape` when the velocity
  closure returns a vector whose length differs from the state.
- `LinearCondField::mse_batch` returns `Error::Domain` for an empty batch or
  dimension instead of an undefined mean.
- `simplex::normalize_simplex` sums in `f64`, so large `f32` inputs no longer
  overflow to infinity.
- `discrete_ctmc`: the `CosineSq` schedule derivative is exactly zero at
  `t = 0` and `t = 1`, and conditional rate construction rejects nonfinite or
  negative rates.

### Changed

- The `sheaf-evals` feature no longer pulls `sheaf`, `serde` or `serde_json`
  into library builds; they are dev-dependencies used only by the gated
  examples, tests and bench.

## [0.1.9] - 2026-07-05

### Added

- `rfm_usgs_full_pipeline_report` can now write a JSON report when
  `FLOWMATCH_REPORT_OUT` is set, recording configs, seeds, metrics, and timing
  breakdowns.

### Changed

- Updated the optional `sheaf-evals` dependency to sheaf 0.1.9.

## [0.1.8] - 2026-06-26

### Added

- `fynch_bregman_energy` example showing `fynch::Shannon` and
  `fynch::SquaredL2` as `logp::BregmanGenerator` energy objectives.

## [0.1.7]

### Added

- `energy::bregman_energy_target`,
  `energy::bregman_energy_matching_target`, and
  `energy::bregman_energy_gradient` for reusing `logp::BregmanGenerator`
  values as flow-matching energy objectives.

## [0.1.6]

### Added

- `VectorField` trait and `flow_drift` helper, moved here from wass. Provides a
  generic abstraction over learned drift functions used by the ODE solvers.
- `pub` visibility on `minibatch_*` pairing functions so external benchmarks
  (`benches/pairing.rs`) can compile against them.

### Changed

- Bumped `rkhs` dependency to 0.2.
- Clippy fixes across examples and tests (no behavioral changes).
- Added `[workspace]` table for standalone builds outside a parent workspace.

## [0.1.5]

(no CHANGELOG entries before 0.1.6 — see git log for prior history)
