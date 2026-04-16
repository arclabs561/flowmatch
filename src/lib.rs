#![warn(missing_docs)]
//! # flowmatch
//!
//! Flow matching as a library primitive.
//!
//! This crate is intentionally small:
//!
//! - it implements **training loops** and **sampling** for flow-matching style models,
//! - it depends on `wass` for OT-shaped coupling primitives (e.g. semidiscrete assignments),
//! - it does not provide a CLI or experiment runner (that belongs in L6 / apps).
//!
//! ## Public invariants (must not change)
//!
//! - **Determinism knobs are explicit**: training/sampling functions take `seed` (or configs do).
//! - **No hidden normalization**: if inputs are normalized, it is stated in the doc comment.
//! - **Backend-agnostic by default**: this crate uses `ndarray` and simple SGD; no GPU framework types
//!   leak through the public API in the default feature set.
//!   - Optional training backends (e.g. `burn`) are **feature-gated**.
//!
//! ## How this maps to “Flow Matching” (papers)
//!
//! The core training objective used here is the standard *conditional flow matching* regression
//! (sample `t`, sample a point on a path `x_t`, regress a vector field `v_theta(x_t, t)`
//! toward a target velocity `u_t`). Concretely:
//!
//! - `sd_fm::train_sd_fm_semidiscrete_linear` uses a **linear interpolation path**
//!   `x_t = (1-t) x_0 + t y_j` and target `u_t = y_j - x_0`.
//! - A semidiscrete “pick `j`” step is provided by `wass::semidiscrete` (potentials + hard assignment),
//!   which acts like a simple coupling / conditioning mechanism.
//!
//! ## References (conceptual anchors; not “implemented fully”)
//!
//! - Lipman et al., *Flow Matching for Generative Modeling* (arXiv:2210.02747):
//!   the canonical FM objective and linear-path baselines.
//! - Lipman et al., *Flow Matching Guide and Code* (arXiv:2412.06264):
//!   a comprehensive reference covering the full design space.
//! - Li et al., *Flow Matching Meets Biology and Life Science: A Survey* (arXiv:2507.17731, 2025):
//!   a taxonomy of variants (CFM/RFM, non-Euclidean, discrete) and a map of applications/tooling.
//! - Gat et al., *Discrete Flow Matching* (NeurIPS 2024):
//!   extending the FM paradigm to discrete data (language, graphs).
//! - Chen & Lipman, *Riemannian Flow Matching on General Geometries* (arXiv:2302.03660):
//!   the foundation for FM on manifolds (like the Poincaré ball in `hyperball`).
//!
//! Related variants that are **not** implemented here (yet):
//!
//! - Dao et al., *Flow Matching in Latent Space* (arXiv:2307.08698) — latent FM + guidance details
//! - Klein et al., *Equivariant Flow Matching* (NeurIPS 2023) — symmetry constraints
//! - Zaghen et al., *Towards Variational Flow Matching on General Geometries* (arXiv:2502.12981, 2025) —
//!   variational objectives with Riemannian Gaussians (RG-VFM).
//!
//! **Applications & Extensions**:
//!
//! - Qin et al., *DeFoG: Discrete Flow Matching for Graph Generation* (arXiv:2410.04263, 2025).
//! - FlowMM: Generating Materials with Riemannian Flow Matching (2024/2025).
//!
//! ## What can change later
//!
//! - The parameterization of vector fields (linear vs MLP vs backend-specific).
//! - ODE integrators (Euler → Heun/RK).
//! - Adding optional Tweedie correction utilities (diffusion-specific).
//!
//! ## Module map
//!
//! - `flow`: [`VectorField`] trait and [`flow_drift`] primitive (canonical home; was `wass::flow`)
//! - `sd_fm`: semidiscrete conditional FM training and sampling
//! - `rfm`: rectified-flow coupling helpers (minibatch OT pairing)
//! - `linear`: simple linear vector-field parameterizations
//! - `ode`: fixed-step ODE integrators (`Euler`, `Heun`)
//! - `ot_cfm`: OT-conditional flow matching (mini-batch OT coupling for CFM training)
//! - `energy`: energy matching (scalar energy whose gradient gives the velocity field)
//! - `metrics`: evaluation metrics (JS divergence, entropic OT cost)
//! - `discrete_ctmc`: CTMC generator scaffolding for discrete FM
//! - `simplex`: simplex utilities for discrete FM variants
//! - `riemannian`: Riemannian FM training (feature-gated: `riemannian`)
//! - `riemannian_ode`: manifold ODE integrators (feature-gated: `riemannian`)
//! - `burn_euclidean`: Burn-backed Euclidean FM training (feature-gated: `burn`)
//! - `burn_sd_fm`: Burn-backed SD-FM/RFM training (feature-gated: `burn`)

pub mod discrete_ctmc;
pub mod energy;
pub mod flow;
pub mod linear;
pub mod metrics;
pub mod ode;
pub mod ot_cfm;
pub mod rfm;
#[cfg(feature = "riemannian")]
pub mod riemannian;
#[cfg(feature = "riemannian")]
pub mod riemannian_ode;
pub mod sd_fm;
pub(crate) mod simplex;

pub use flow::{flow_drift, VectorField};

#[cfg(feature = "burn")]
pub mod burn_euclidean;

#[cfg(feature = "burn")]
pub mod burn_sd_fm;

/// flowmatch error variants.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Array shape or dimension mismatch.
    #[error("shape mismatch: {0}")]
    Shape(&'static str),
    /// Value outside the valid domain (e.g., `t` not in `[0, 1]`).
    #[error("domain error: {0}")]
    Domain(&'static str),
}

/// Convenience alias for `std::result::Result<T, flowmatch::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
