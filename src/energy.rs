//! Energy matching for flow matching.
//!
//! Energy matching (Akhound-Sadegh et al. 2025) unifies flow matching and energy-based models.
//! Instead of regressing a velocity field directly, a scalar energy function `E(x, t)` is
//! learned whose gradient gives the velocity: `v(x, t) = -grad_x E(x, t)`.
//!
//! For the conditional Gaussian path `p_t(x | x_0, x_1)`, the energy target is:
//!
//! $$
//! E(x, t) = -\frac{1}{2} \frac{\|x - \mu_t\|^2}{\sigma_t^2}
//! $$
//!
//! where `mu_t = (1-t)*x_0 + t*x_1` and `sigma_t` is a noise schedule parameter.
//!
//! The velocity field is recovered as `v = -grad_x E = (x - mu_t) / sigma_t^2`,
//! which for `sigma_t -> 0` reduces to the standard conditional flow matching target.

use crate::{Error, Result};
use ndarray::{Array1, ArrayView1};

/// Configuration for energy matching.
#[derive(Debug, Clone)]
pub struct EnergyMatchingConfig {
    /// Noise scale for the conditional Gaussian path. Controls the width of the
    /// conditional distribution `p_t(x | x_0, x_1)`. Smaller values produce
    /// sharper energy landscapes; must be positive and finite.
    pub sigma: f32,
    /// Number of samples for Monte Carlo gradient estimation (unused in the
    /// closed-form energy target, reserved for stochastic variants).
    pub num_samples: usize,
}

impl Default for EnergyMatchingConfig {
    fn default() -> Self {
        Self {
            sigma: 1e-2,
            num_samples: 1,
        }
    }
}

/// Compute the energy matching target for a sample pair at time `t`.
///
/// For the linear interpolation path with Gaussian noise:
///
/// $$
/// E(x, t) = -\frac{1}{2} \frac{\|x - \mu_t\|^2}{\sigma^2}
/// $$
///
/// where `mu_t = (1 - t) * x_0 + t * x_1`.
///
/// The evaluation point `x` is the interpolated sample (possibly with added noise).
///
/// # Errors
///
/// Returns [`Error::Shape`] if `x0`, `x1`, and `x` have different lengths.
/// Returns [`Error::Domain`] if `t` is outside `[0, 1]` or `sigma` is not positive/finite.
pub fn energy_matching_target(
    x: &ArrayView1<f32>,
    x0: &ArrayView1<f32>,
    x1: &ArrayView1<f32>,
    t: f32,
    config: &EnergyMatchingConfig,
) -> Result<f32> {
    let d = x0.len();
    if x1.len() != d || x.len() != d {
        return Err(Error::Shape("x, x0, and x1 must have the same length"));
    }
    if !(0.0..=1.0).contains(&t) {
        return Err(Error::Domain("t must be in [0, 1]"));
    }
    validate_sigma(config.sigma)?;

    let sigma_sq = config.sigma * config.sigma;

    // mu_t = (1 - t) * x0 + t * x1
    // E = -0.5 * ||x - mu_t||^2 / sigma^2
    let mut dist_sq = 0.0f64;
    for i in 0..d {
        let mu_i = (1.0 - t) * x0[i] + t * x1[i];
        let diff = (x[i] - mu_i) as f64;
        dist_sq += diff * diff;
    }

    Ok((-0.5 * dist_sq / sigma_sq as f64) as f32)
}

/// MSE loss between predicted and target energy values.
///
/// $$
/// L = \frac{1}{n} \sum_{i=1}^{n} (E_\theta(x_i, t_i) - E^*(x_i, t_i))^2
/// $$
pub fn energy_matching_loss(predicted: &[f32], target: &[f32]) -> Result<f32> {
    if predicted.len() != target.len() {
        return Err(Error::Shape(
            "predicted and target must have the same length",
        ));
    }
    if predicted.is_empty() {
        return Err(Error::Shape("batch must be non-empty"));
    }

    let n = predicted.len();
    let mut sum = 0.0f64;
    for i in 0..n {
        let diff = (predicted[i] - target[i]) as f64;
        sum += diff * diff;
    }
    Ok((sum / n as f64) as f32)
}

/// Convert an energy gradient to a velocity field.
///
/// The velocity is the negative gradient of the energy:
///
/// $$
/// v(x, t) = -\nabla_x E(x, t)
/// $$
///
/// This function simply negates the input gradient.
pub fn score_from_energy(energy_grad: &ArrayView1<f32>) -> Array1<f32> {
    let mut v = Array1::<f32>::zeros(energy_grad.len());
    for i in 0..energy_grad.len() {
        v[i] = -energy_grad[i];
    }
    v
}

/// Compute the analytical energy gradient at point `x` for the conditional Gaussian path.
///
/// $$
/// \nabla_x E = -\frac{x - \mu_t}{\sigma^2}
/// $$
///
/// This is useful for verifying that `score_from_energy` produces the correct velocity.
pub fn energy_gradient(
    x: &ArrayView1<f32>,
    x0: &ArrayView1<f32>,
    x1: &ArrayView1<f32>,
    t: f32,
    config: &EnergyMatchingConfig,
) -> Result<Array1<f32>> {
    let d = x0.len();
    if x1.len() != d || x.len() != d {
        return Err(Error::Shape("x, x0, and x1 must have the same length"));
    }
    if !(0.0..=1.0).contains(&t) {
        return Err(Error::Domain("t must be in [0, 1]"));
    }
    validate_sigma(config.sigma)?;

    let sigma_sq = config.sigma * config.sigma;
    let mut grad = Array1::<f32>::zeros(d);
    for i in 0..d {
        let mu_i = (1.0 - t) * x0[i] + t * x1[i];
        grad[i] = -(x[i] - mu_i) / sigma_sq;
    }
    Ok(grad)
}

fn validate_sigma(sigma: f32) -> Result<()> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error::Domain("sigma must be positive and finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn arr(v: &[f32]) -> Array1<f32> {
        Array1::from_vec(v.to_vec())
    }

    #[test]
    fn energy_at_t0_peaks_near_x0() {
        // At t=0, mu_t = x0. Energy at x=x0 should be 0 (maximum).
        // Energy at x far from x0 should be negative.
        let x0 = arr(&[1.0, 2.0, 3.0]);
        let x1 = arr(&[4.0, 5.0, 6.0]);
        let config = EnergyMatchingConfig {
            sigma: 0.1,
            num_samples: 1,
        };

        let e_at_x0 =
            energy_matching_target(&x0.view(), &x0.view(), &x1.view(), 0.0, &config).unwrap();
        assert!(
            e_at_x0.abs() < 1e-6,
            "energy at x=mu_0=x0 should be ~0, got {e_at_x0}"
        );

        let x_far = arr(&[10.0, 10.0, 10.0]);
        let e_far =
            energy_matching_target(&x_far.view(), &x0.view(), &x1.view(), 0.0, &config).unwrap();
        assert!(
            e_far < e_at_x0,
            "energy far from x0 should be lower (more negative): {e_far} vs {e_at_x0}"
        );
    }

    #[test]
    fn energy_at_t1_peaks_near_x1() {
        // At t=1, mu_t = x1. Energy at x=x1 should be 0 (maximum).
        let x0 = arr(&[1.0, 2.0, 3.0]);
        let x1 = arr(&[4.0, 5.0, 6.0]);
        let config = EnergyMatchingConfig {
            sigma: 0.1,
            num_samples: 1,
        };

        let e_at_x1 =
            energy_matching_target(&x1.view(), &x0.view(), &x1.view(), 1.0, &config).unwrap();
        assert!(
            e_at_x1.abs() < 1e-6,
            "energy at x=mu_1=x1 should be ~0, got {e_at_x1}"
        );

        let x_far = arr(&[10.0, 10.0, 10.0]);
        let e_far =
            energy_matching_target(&x_far.view(), &x0.view(), &x1.view(), 1.0, &config).unwrap();
        assert!(
            e_far < e_at_x1,
            "energy far from x1 should be lower (more negative): {e_far} vs {e_at_x1}"
        );
    }

    #[test]
    fn score_from_energy_negates_gradient() {
        let grad = arr(&[1.0, -2.0, 3.5]);
        let v = score_from_energy(&grad.view());
        assert!((v[0] - (-1.0)).abs() < 1e-7);
        assert!((v[1] - 2.0).abs() < 1e-7);
        assert!((v[2] - (-3.5)).abs() < 1e-7);
    }

    #[test]
    fn velocity_from_energy_gradient_matches_cfm_target() {
        // The velocity v = -grad_E should point from x toward mu_t,
        // scaled by 1/sigma^2. For the linear path, the CFM target velocity
        // is u_t = x1 - x0. At x = mu_t, grad_E = 0 so v = 0.
        // At x = mu_t + delta, v = delta / sigma^2 (pointing back toward mu_t).
        let x0 = arr(&[0.0, 0.0]);
        let x1 = arr(&[2.0, 4.0]);
        let t = 0.5;
        let config = EnergyMatchingConfig {
            sigma: 1.0,
            num_samples: 1,
        };

        // mu_t = [1.0, 2.0]
        let delta = arr(&[0.1, -0.2]);
        let x = arr(&[1.1, 1.8]); // mu_t + delta

        let grad = energy_gradient(&x.view(), &x0.view(), &x1.view(), t, &config).unwrap();
        let v = score_from_energy(&grad.view());

        // v should equal delta / sigma^2 = delta (since sigma=1)
        for i in 0..2 {
            let expected = delta[i]; // / 1.0^2
            assert!(
                (v[i] - expected).abs() < 1e-6,
                "velocity[{i}] = {}, expected {expected}",
                v[i]
            );
        }
    }

    #[test]
    fn energy_matching_loss_zero_for_equal() {
        let vals = vec![1.0f32, 2.0, 3.0];
        let loss = energy_matching_loss(&vals, &vals).unwrap();
        assert!(
            loss.abs() < 1e-7,
            "loss should be 0 for equal inputs, got {loss}"
        );
    }

    #[test]
    fn energy_matching_loss_correct_mse() {
        let pred = vec![1.0f32, 2.0, 3.0];
        let tgt = vec![2.0f32, 2.0, 1.0];
        let loss = energy_matching_loss(&pred, &tgt).unwrap();
        // (1 + 0 + 4) / 3 = 5/3
        let expected = 5.0 / 3.0;
        assert!(
            (loss - expected).abs() < 1e-5,
            "expected {expected}, got {loss}"
        );
    }

    #[test]
    fn shape_mismatch_errors() {
        let x0 = arr(&[1.0, 2.0]);
        let x1 = arr(&[3.0, 4.0, 5.0]);
        let x = arr(&[1.0, 2.0]);
        let config = EnergyMatchingConfig::default();

        assert!(energy_matching_target(&x.view(), &x0.view(), &x1.view(), 0.5, &config).is_err());
        assert!(energy_gradient(&x.view(), &x0.view(), &x1.view(), 0.5, &config).is_err());
    }

    #[test]
    fn domain_errors() {
        let x0 = arr(&[1.0, 2.0]);
        let x1 = arr(&[3.0, 4.0]);
        let x = arr(&[2.0, 3.0]);

        // t out of range
        let config = EnergyMatchingConfig::default();
        assert!(energy_matching_target(&x.view(), &x0.view(), &x1.view(), -0.1, &config).is_err());
        assert!(energy_matching_target(&x.view(), &x0.view(), &x1.view(), 1.1, &config).is_err());

        // bad sigma
        let bad_config = EnergyMatchingConfig {
            sigma: 0.0,
            num_samples: 1,
        };
        assert!(
            energy_matching_target(&x.view(), &x0.view(), &x1.view(), 0.5, &bad_config).is_err()
        );

        // loss shape mismatch
        assert!(energy_matching_loss(&[1.0], &[1.0, 2.0]).is_err());
        assert!(energy_matching_loss(&[], &[]).is_err());
    }

    #[test]
    fn energy_is_monotone_in_distance() {
        // Energy should decrease (become more negative) as x moves away from mu_t.
        let x0 = arr(&[0.0]);
        let x1 = arr(&[1.0]);
        let config = EnergyMatchingConfig {
            sigma: 0.5,
            num_samples: 1,
        };
        let t = 0.5; // mu_t = 0.5

        let x_near = arr(&[0.5]);
        let x_mid = arr(&[1.0]);
        let x_far = arr(&[2.0]);

        let e_near =
            energy_matching_target(&x_near.view(), &x0.view(), &x1.view(), t, &config).unwrap();
        let e_mid =
            energy_matching_target(&x_mid.view(), &x0.view(), &x1.view(), t, &config).unwrap();
        let e_far =
            energy_matching_target(&x_far.view(), &x0.view(), &x1.view(), t, &config).unwrap();

        assert!(
            e_near > e_mid,
            "closer point should have higher energy: {e_near} vs {e_mid}"
        );
        assert!(
            e_mid > e_far,
            "medium point should have higher energy than far: {e_mid} vs {e_far}"
        );
    }
}
