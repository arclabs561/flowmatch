# Changelog

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
