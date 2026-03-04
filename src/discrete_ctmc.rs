//! Discrete FM (CTMC-based) scaffolding.
//!
//! CTMC-based "discrete flow matching" methods learn (or use) a continuous-time Markov chain
//! whose generator \(Q(t)\) defines the evolution of a categorical distribution \(p(t)\).
//!
//! This module provides:
//! - a generator validation contract
//! - a minimal probability evolution step (forward Euler)
//! - **interpolation schedules** for discrete probability paths (Gat et al. 2024, DFM)
//! - **conditional probability paths** `p_t(x | x_0, x_1)` between source/target one-hots
//! - a **conditional rate matrix** builder for the DFM CTMC
//!
//! ## References
//!
//! - Gat et al., *Discrete Flow Matching* (NeurIPS 2024):
//!   The `kappa(t) = 1 - cos(pi*t/2)` schedule and the conditional rate matrix construction.
//! - Flowfusion.jl (`InterpolatingDiscreteFlow`):
//!   Julia reference implementing the same cosine schedule.

use crate::{Error, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ---------------------------------------------------------------------------
// Interpolation schedules for discrete probability paths
// ---------------------------------------------------------------------------

/// Interpolation schedule for the discrete probability path.
///
/// Given a schedule `kappa: [0,1] -> [0,1]` with `kappa(0)=0, kappa(1)=1`, the conditional
/// probability path is:
///
/// ```text
/// p_t(x | x_0, x_1) = (1 - kappa(t)) * delta_{x_0}(x) + kappa(t) * delta_{x_1}(x)
/// ```
///
/// For `x_0 != x_1`, this linearly interpolates the one-hot distributions.
/// For `x_0 == x_1`, `p_t = delta_{x_0}` for all t.
#[derive(Debug, Clone, Copy)]
pub enum DiscreteSchedule {
    /// Linear: `kappa(t) = t`. Simple but not recommended (Gat et al. note cosine is better).
    Linear,
    /// Cosine-squared: `kappa(t) = sin^2(pi * t / 2)`.
    ///
    /// This is the schedule from Gat et al. (2024, NeurIPS):
    /// `kappa(t) = sin^2(pi*t/2) = (1 - cos(pi*t)) / 2`.
    ///
    /// It starts slow near t=0, accelerates through mid-range, and decelerates near t=1.
    /// The derivative `kappa'(t) = (pi/2) * sin(pi*t)` vanishes at both endpoints,
    /// which avoids the `1/(1-t)` singularity of the linear schedule.
    CosineSq,
    /// Flowfusion-style cosine: `kappa(t) = 1 - cos(pi * t / 2)`.
    ///
    /// Used by Flowfusion.jl's `InterpolatingDiscreteFlow`. Starts slow near t=0 but
    /// reaches full velocity at t=1 (derivative does not vanish at t=1).
    CosineHalf,
}

impl DiscreteSchedule {
    /// Evaluate the schedule: `kappa(t)`.
    pub fn kappa(&self, t: f32) -> f32 {
        debug_assert!((0.0..=1.0).contains(&t), "t must be in [0, 1], got {t}");
        match self {
            Self::Linear => t,
            Self::CosineSq => {
                let s = (std::f32::consts::FRAC_PI_2 * t).sin();
                s * s
            }
            Self::CosineHalf => 1.0 - (std::f32::consts::FRAC_PI_2 * t).cos(),
        }
    }

    /// Derivative of the schedule: `kappa'(t)`.
    ///
    /// Used for the probability velocity / conditional rate matrix.
    pub fn kappa_dot(&self, t: f32) -> f32 {
        debug_assert!((0.0..=1.0).contains(&t), "t must be in [0, 1], got {t}");
        match self {
            Self::Linear => 1.0,
            // d/dt sin^2(pi*t/2) = (pi/2) * sin(pi*t)
            Self::CosineSq => std::f32::consts::FRAC_PI_2 * (std::f32::consts::PI * t).sin(),
            // d/dt (1 - cos(pi*t/2)) = (pi/2) * sin(pi*t/2)
            Self::CosineHalf => {
                std::f32::consts::FRAC_PI_2 * (std::f32::consts::FRAC_PI_2 * t).sin()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Conditional probability path
// ---------------------------------------------------------------------------

/// Compute the conditional probability vector `p_t(x | x_0, x_1)` over `k` states.
///
/// Returns a length-`k` probability vector where:
/// - `p_t[x_0] += (1 - kappa(t))`
/// - `p_t[x_1] += kappa(t)`
///
/// When `x_0 == x_1`, the result is a one-hot on that state.
pub fn conditional_probability_path(
    schedule: DiscreteSchedule,
    t: f32,
    x0: usize,
    x1: usize,
    k: usize,
) -> Result<Array1<f32>> {
    if x0 >= k || x1 >= k {
        return Err(Error::Domain("x0 and x1 must be < k"));
    }
    if !t.is_finite() || !(0.0..=1.0).contains(&t) {
        return Err(Error::Domain("t must be in [0, 1]"));
    }
    let kap = schedule.kappa(t);
    let mut p = Array1::zeros(k);
    p[x0] += 1.0 - kap;
    p[x1] += kap;
    Ok(p)
}

// ---------------------------------------------------------------------------
// Conditional rate matrix (DFM)
// ---------------------------------------------------------------------------

/// Build the conditional rate matrix `R_t(x' | x; x_0, x_1)` for a single (x_0, x_1) pair.
///
/// From Gat et al. (2024), the conditional rate for transitioning from state `x` to `x'` is:
///
/// ```text
/// R_t(x' | x) = kappa'(t) / (1 - kappa(t)) * [x' == x_1]   if x == x_0 and x_0 != x_1
///             = 0                                              otherwise
/// ```
///
/// This is a valid CTMC generator: off-diagonal entries are nonneg, rows sum to 0.
///
/// The `eps` parameter clamps `(1 - kappa(t))` away from zero to avoid division by zero
/// near `t=1`. A typical value is `1e-5`.
pub fn conditional_rate_matrix(
    schedule: DiscreteSchedule,
    t: f32,
    x0: usize,
    x1: usize,
    k: usize,
    eps: f32,
) -> Result<Array2<f32>> {
    if x0 >= k || x1 >= k {
        return Err(Error::Domain("x0 and x1 must be < k"));
    }
    if !t.is_finite() || !(0.0..=1.0).contains(&t) {
        return Err(Error::Domain("t must be in [0, 1]"));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Error::Domain("eps must be finite and > 0"));
    }

    let mut r = Array2::<f32>::zeros((k, k));

    if x0 == x1 {
        // Already at target — no transitions needed.
        return Ok(r);
    }

    let kap = schedule.kappa(t);
    let kap_dot = schedule.kappa_dot(t);
    let denom = (1.0 - kap).max(eps);
    let rate = kap_dot / denom;

    // Only the row for state x0 has a nonzero off-diagonal entry (to x1).
    r[[x0, x1]] = rate;
    r[[x0, x0]] = -rate;

    Ok(r)
}

// ---------------------------------------------------------------------------
// CTMC generator (time-homogeneous, for generic use)
// ---------------------------------------------------------------------------

/// A time-homogeneous CTMC generator matrix \(Q\).
///
/// Convention: row-stochastic evolution on the left:
/// \[
/// \frac{dp}{dt} = p Q,
/// \]
/// where `p` is a row vector (probabilities over states).
#[derive(Debug, Clone)]
pub struct CtmcGenerator {
    pub q: Array2<f32>,
}

impl CtmcGenerator {
    /// Validate the CTMC generator constraints:
    /// - off-diagonal entries are nonnegative
    /// - each row sums to ~0 (within `tol`)
    pub fn validate(&self, tol: f32) -> Result<()> {
        validate_generator(&self.q.view(), tol)
    }

    /// Forward Euler step: `p_next = p + dt * p Q`.
    ///
    /// This does not project/renormalize; caller can check invariants.
    pub fn step_euler(&self, p: &ArrayView1<f32>, dt: f32) -> Result<Array1<f32>> {
        if p.len() != self.q.nrows() {
            return Err(Error::Shape("p length must match Q dimension"));
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(Error::Domain("dt must be finite and >= 0"));
        }
        let dp = p.dot(&self.q);
        let mut out = p.to_owned();
        for i in 0..out.len() {
            out[i] += dt * dp[i];
        }
        Ok(out)
    }
}

/// Check that `q` is a valid CTMC rate matrix (non-negative off-diagonals, rows sum to zero).
pub fn validate_generator(q: &ArrayView2<f32>, tol: f32) -> Result<()> {
    let n = q.nrows();
    if q.ncols() != n {
        return Err(Error::Shape("Q must be square"));
    }
    if n == 0 {
        return Err(Error::Domain("Q must be non-empty"));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(Error::Domain("tol must be finite and >= 0"));
    }
    if q.iter().any(|&x| !x.is_finite()) {
        return Err(Error::Domain("Q contains non-finite values"));
    }

    for i in 0..n {
        let mut row_sum = 0.0f64;
        for j in 0..n {
            let v = q[[i, j]];
            if i != j && v < -tol {
                return Err(Error::Domain("off-diagonal rates must be nonnegative"));
            }
            row_sum += v as f64;
        }
        if (row_sum as f32).abs() > tol {
            return Err(Error::Domain("each row of Q must sum to 0 (within tol)"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use proptest::prelude::*;

    #[test]
    fn ctmc_generator_preserves_total_mass_to_first_order() {
        // Two-state CTMC: 0 -> 1 with rate a, 1 -> 0 with rate b.
        let a = 2.0f32;
        let b = 3.0f32;
        let q = array![[-a, a], [b, -b]];
        let gen = CtmcGenerator { q };
        gen.validate(1e-6).unwrap();

        let p0 = array![0.4f32, 0.6f32];
        let p1 = gen.step_euler(&p0.view(), 1e-3).unwrap();

        let s0 = p0.sum();
        let s1 = p1.sum();
        assert!((s0 - s1).abs() < 1e-5, "mass drift too large: {s0} vs {s1}");
    }

    // --- Schedule tests ---

    #[test]
    fn cosine_schedule_boundary_values() {
        let s = DiscreteSchedule::CosineSq;
        assert!((s.kappa(0.0)).abs() < 1e-7, "kappa(0) should be 0");
        assert!((s.kappa(1.0) - 1.0).abs() < 1e-6, "kappa(1) should be 1");
    }

    #[test]
    fn linear_schedule_boundary_values() {
        let s = DiscreteSchedule::Linear;
        assert_eq!(s.kappa(0.0), 0.0);
        assert_eq!(s.kappa(1.0), 1.0);
        assert_eq!(s.kappa(0.5), 0.5);
    }

    #[test]
    fn cosine_schedule_is_monotone_increasing() {
        let s = DiscreteSchedule::CosineSq;
        let steps = 100;
        let mut prev = 0.0f32;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let k = s.kappa(t);
            assert!(
                k >= prev - 1e-7,
                "kappa not monotone at t={t}: {prev} -> {k}"
            );
            prev = k;
        }
    }

    #[test]
    fn cosine_kappa_dot_is_nonneg() {
        let s = DiscreteSchedule::CosineSq;
        let steps = 100;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let kd = s.kappa_dot(t);
            assert!(kd >= -1e-6, "kappa_dot negative at t={t}: {kd}");
        }
    }

    #[test]
    fn cosine_half_schedule_boundary_values() {
        let s = DiscreteSchedule::CosineHalf;
        assert!((s.kappa(0.0)).abs() < 1e-7, "kappa(0) should be 0");
        assert!((s.kappa(1.0) - 1.0).abs() < 1e-6, "kappa(1) should be 1");
    }

    #[test]
    fn cosine_sq_and_half_differ_at_midpoint() {
        // sin^2(pi/4) = 0.5, but 1 - cos(pi/4) ~= 0.293
        let sq = DiscreteSchedule::CosineSq.kappa(0.5);
        let half = DiscreteSchedule::CosineHalf.kappa(0.5);
        assert!(
            (sq - 0.5).abs() < 1e-6,
            "CosineSq(0.5) should be 0.5, got {sq}"
        );
        assert!(
            (half - 0.5).abs() > 0.1,
            "CosineHalf(0.5) should differ from 0.5, got {half}"
        );
    }

    // --- Conditional probability path tests ---

    #[test]
    fn conditional_path_boundary_t0_is_source() {
        let p = conditional_probability_path(DiscreteSchedule::CosineSq, 0.0, 0, 2, 4).unwrap();
        assert!((p[0] - 1.0).abs() < 1e-6);
        assert!(p[2].abs() < 1e-6);
    }

    #[test]
    fn conditional_path_boundary_t1_is_target() {
        let p = conditional_probability_path(DiscreteSchedule::CosineSq, 1.0, 0, 2, 4).unwrap();
        assert!(p[0].abs() < 1e-6);
        assert!((p[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn conditional_path_same_state_is_one_hot() {
        // When x0 == x1, the path is always a one-hot on that state.
        let p = conditional_probability_path(DiscreteSchedule::CosineSq, 0.5, 1, 1, 3).unwrap();
        assert!((p[1] - 1.0).abs() < 1e-7);
        assert!(p[0].abs() < 1e-7);
        assert!(p[2].abs() < 1e-7);
    }

    #[test]
    fn conditional_path_sums_to_one() {
        let p = conditional_probability_path(DiscreteSchedule::CosineSq, 0.37, 1, 3, 5).unwrap();
        assert!((p.sum() - 1.0).abs() < 1e-6, "sum={}", p.sum());
    }

    // --- Conditional rate matrix tests ---

    #[test]
    fn conditional_rate_matrix_is_valid_generator() {
        let r = conditional_rate_matrix(DiscreteSchedule::CosineSq, 0.3, 0, 2, 4, 1e-5).unwrap();
        let gen = CtmcGenerator { q: r };
        gen.validate(1e-5).unwrap();
    }

    #[test]
    fn conditional_rate_matrix_same_state_is_zero() {
        let r = conditional_rate_matrix(DiscreteSchedule::CosineSq, 0.5, 1, 1, 3, 1e-5).unwrap();
        for &v in r.iter() {
            assert!(
                v.abs() < 1e-10,
                "expected zero matrix when x0 == x1, got {v}"
            );
        }
    }

    #[test]
    fn conditional_rate_only_x0_row_nonzero() {
        let k = 5;
        let x0 = 1;
        let x1 = 3;
        let r = conditional_rate_matrix(DiscreteSchedule::CosineSq, 0.4, x0, x1, k, 1e-5).unwrap();

        // All rows except x0 should be zero.
        for i in 0..k {
            if i == x0 {
                continue;
            }
            for j in 0..k {
                assert!(r[[i, j]].abs() < 1e-10, "row {i} should be zero");
            }
        }
        // Row x0 should have: negative diagonal, positive entry at x1, zero elsewhere.
        assert!(r[[x0, x1]] > 0.0, "rate x0->x1 should be positive");
        assert!(
            (r[[x0, x0]] + r[[x0, x1]]).abs() < 1e-6,
            "row must sum to 0"
        );
    }

    // --- Proptest: schedule monotonicity + path validity ---

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_cosine_kappa_in_unit_interval(t in 0.0f32..=1.0f32) {
            let k = DiscreteSchedule::CosineSq.kappa(t);
            prop_assert!((-1e-7..=1.0 + 1e-7).contains(&k), "kappa({t}) = {k} out of [0,1]");
        }

        #[test]
        fn prop_conditional_path_always_valid_distribution(
            t in 0.0f32..=1.0f32,
            x0 in 0usize..16,
            x1 in 0usize..16,
        ) {
            let k = 16;
            let p = conditional_probability_path(DiscreteSchedule::CosineSq, t, x0, x1, k).unwrap();

            // All entries nonneg.
            for &v in p.iter() {
                prop_assert!(v >= -1e-7, "negative probability: {v}");
            }
            // Sum to 1.
            prop_assert!((p.sum() - 1.0).abs() < 1e-5, "sum = {}", p.sum());
        }

        #[test]
        fn prop_conditional_rate_matrix_is_valid_generator(
            t in 0.01f32..0.99f32,
            x0 in 0usize..8,
            x1 in 0usize..8,
        ) {
            let k = 8;
            let r = conditional_rate_matrix(DiscreteSchedule::CosineSq, t, x0, x1, k, 1e-5).unwrap();
            let gen = CtmcGenerator { q: r };
            gen.validate(1e-4).map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
        }

        #[test]
        fn prop_euler_step_with_conditional_rate_moves_toward_target(
            t in 0.01f32..0.5f32,
        ) {
            let k = 4;
            let x0 = 0usize;
            let x1 = 2usize;
            let dt = 0.001f32;

            // Start at p = one-hot on x0.
            let p = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
            let r = conditional_rate_matrix(DiscreteSchedule::CosineSq, t, x0, x1, k, 1e-5).unwrap();
            let gen = CtmcGenerator { q: r };
            let p_next = gen.step_euler(&p.view(), dt).unwrap();

            // After one step, probability should have moved toward x1.
            prop_assert!(p_next[x1] > 0.0, "x1 prob should increase: {}", p_next[x1]);
            prop_assert!(p_next[x0] < 1.0, "x0 prob should decrease: {}", p_next[x0]);
        }
    }
}
