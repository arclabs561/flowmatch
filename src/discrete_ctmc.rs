//! Discrete FM (CTMC-based) scaffolding.
//!
//! CTMC-based “discrete flow matching” methods learn (or use) a continuous-time Markov chain
//! whose generator \(Q(t)\) defines the evolution of a categorical distribution \(p(t)\).
//!
//! This module does **not** implement a full learning objective; it provides:
//! - a generator validation contract
//! - a minimal probability evolution step

use crate::{Error, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

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
        if !(dt >= 0.0) || !dt.is_finite() {
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

pub fn validate_generator(q: &ArrayView2<f32>, tol: f32) -> Result<()> {
    let n = q.nrows();
    if q.ncols() != n {
        return Err(Error::Shape("Q must be square"));
    }
    if n == 0 {
        return Err(Error::Domain("Q must be non-empty"));
    }
    if !(tol >= 0.0) || !tol.is_finite() {
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
}
