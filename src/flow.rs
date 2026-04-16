//! Flow / drift primitives.
//!
//! `VectorField` and `flow_drift` originated in `wass::flow` and have been
//! promoted here as the canonical location. The `wass` re-exports remain but
//! are deprecated.
//!
//! # References
//!
//! - Pooladian et al. (2024). "Neural OT with Lagrangian Costs" -- connects
//!   flow/drift to Lagrangian OT where transport follows a least-action principle.

use ndarray::{Array1, ArrayView1};

/// A vector field representing a continuous drift in a (latent) space.
pub trait VectorField {
    /// Evaluate the velocity at point `x` and time `t`.
    fn velocity(&self, x: &ArrayView1<f64>, t: f64) -> Array1<f64>;
}

/// Computes the drift between two points.
///
/// \[
/// v = \frac{\text{target} - \text{source}}{\Delta t}
/// \]
///
/// Panics if `source.len() != target.len()` or if `dt == 0`.
pub fn flow_drift(source: &[f64], target: &[f64], dt: f64) -> Vec<f64> {
    assert_eq!(source.len(), target.len(), "dimension mismatch");
    assert!(dt != 0.0, "dt must be non-zero");
    source
        .iter()
        .zip(target)
        .map(|(&s, &t)| (t - s) / dt)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_unit_step() {
        let v = flow_drift(&[0.0, 0.0], &[1.0, 2.0], 1.0);
        assert_eq!(v, vec![1.0, 2.0]);
    }

    #[test]
    fn drift_half_step() {
        let v = flow_drift(&[0.0], &[1.0], 0.5);
        assert!((v[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn drift_same_point_is_zero() {
        let v = flow_drift(&[3.0, 4.0], &[3.0, 4.0], 1.0);
        assert!(v.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    #[should_panic(expected = "dt must be non-zero")]
    fn drift_panics_on_zero_dt() {
        flow_drift(&[0.0], &[1.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn drift_panics_on_dim_mismatch() {
        flow_drift(&[0.0, 1.0], &[1.0], 1.0);
    }
}
