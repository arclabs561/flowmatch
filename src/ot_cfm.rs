//! OT-conditional flow matching (OT-CFM).
//!
//! Standard CFM pairs source `x0` and target `x1` samples independently (random permutation).
//! OT-CFM computes a mini-batch optimal transport coupling between them first, producing
//! straighter flow paths and faster training convergence.
//!
//! Reference: Tong et al. 2023, "Improving and Generalizing Flow-Based Generative Models
//! with Minibatch Optimal Transport" (TorchCFM).
//!
//! This module provides:
//! - [`ot_cfm_coupling`]: compute OT-coupled pairing indices from source/target batches.
//! - [`ot_cfm_training_step`]: compute interpolated points and target velocities for a
//!   training step, using OT coupling.

use crate::{Error, Result};
use ndarray::{Array2, ArrayView2};

/// Configuration for OT-conditional flow matching.
#[derive(Debug, Clone)]
pub struct OtCfmConfig {
    /// Sinkhorn entropic regularization (larger = smoother coupling, smaller = sharper).
    pub reg: f32,
    /// Maximum Sinkhorn iterations.
    pub max_sinkhorn_iter: usize,
    /// Convergence tolerance for Sinkhorn marginal error.
    pub sinkhorn_tol: f32,
}

impl Default for OtCfmConfig {
    fn default() -> Self {
        Self {
            reg: 1.0,
            max_sinkhorn_iter: 6_000,
            sinkhorn_tol: 2e-3,
        }
    }
}

/// Training targets produced by [`ot_cfm_training_step`].
#[derive(Debug)]
pub struct OtCfmTargets {
    /// Interpolated points: `x_t[i] = (1-t[i]) * x0[i] + t[i] * x1[coupling[i]]`.
    pub x_t: Array2<f32>,
    /// Target velocities: `u_t[i] = x1[coupling[i]] - x0[i]`.
    pub u_t: Array2<f32>,
    /// Pairing indices: `coupling[i] = j` means source `i` is paired with target `j`.
    pub coupling: Vec<usize>,
}

/// Compute OT-coupled pairing indices from source and target batches.
///
/// Returns `coupling` where `coupling[i] = j` means `source[i]` is paired with `target[j]`.
/// The result is a valid permutation (each source paired with exactly one target).
///
/// Uses Sinkhorn OT plan + greedy matching, identical to
/// [`rfm::minibatch_ot_greedy_pairing`](crate::rfm::minibatch_ot_greedy_pairing).
pub fn ot_cfm_coupling(
    source: &ArrayView2<f32>,
    target: &ArrayView2<f32>,
    config: &OtCfmConfig,
) -> Result<Vec<usize>> {
    validate_config(config)?;
    crate::rfm::minibatch_ot_greedy_pairing(
        source,
        target,
        config.reg,
        config.max_sinkhorn_iter,
        config.sinkhorn_tol,
    )
}

/// Compute OT-CFM training targets for one mini-batch.
///
/// Given source `x0` (batch_size x dim), target `x1` (batch_size x dim), and per-sample
/// timesteps `t` (length batch_size, each in `[0, 1]`):
///
/// 1. Compute the OT coupling between `x0` and `x1`.
/// 2. Reorder targets according to the coupling.
/// 3. Interpolate: `x_t[i] = (1 - t[i]) * x0[i] + t[i] * x1[coupling[i]]`.
/// 4. Target velocity: `u_t[i] = x1[coupling[i]] - x0[i]` (linear path).
pub fn ot_cfm_training_step(
    x0: &ArrayView2<f32>,
    x1: &ArrayView2<f32>,
    t: &[f32],
    config: &OtCfmConfig,
) -> Result<OtCfmTargets> {
    let n = x0.nrows();
    let d = x0.ncols();

    if x1.nrows() != n {
        return Err(Error::Shape("x0 and x1 must have same number of rows"));
    }
    if x1.ncols() != d {
        return Err(Error::Shape("x0 and x1 must have same dimension"));
    }
    if t.len() != n {
        return Err(Error::Shape("t must have length equal to batch size"));
    }
    for &ti in t {
        if !(0.0..=1.0).contains(&ti) {
            return Err(Error::Domain("each t[i] must be in [0, 1]"));
        }
    }

    let coupling = ot_cfm_coupling(&x0.view(), &x1.view(), config)?;

    let mut x_t = Array2::<f32>::zeros((n, d));
    let mut u_t = Array2::<f32>::zeros((n, d));

    for i in 0..n {
        let j = coupling[i];
        let ti = t[i];
        for k in 0..d {
            let x0_ik = x0[[i, k]];
            let x1_jk = x1[[j, k]];
            u_t[[i, k]] = x1_jk - x0_ik;
            x_t[[i, k]] = (1.0 - ti) * x0_ik + ti * x1_jk;
        }
    }

    Ok(OtCfmTargets { x_t, u_t, coupling })
}

fn validate_config(config: &OtCfmConfig) -> Result<()> {
    if !config.reg.is_finite() || config.reg <= 0.0 {
        return Err(Error::Domain("reg must be positive and finite"));
    }
    if config.max_sinkhorn_iter == 0 {
        return Err(Error::Domain("max_sinkhorn_iter must be >= 1"));
    }
    if !config.sinkhorn_tol.is_finite() || config.sinkhorn_tol <= 0.0 {
        return Err(Error::Domain("sinkhorn_tol must be positive and finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_test_data(n: usize, d: usize, seed: u64) -> (Array2<f32>, Array2<f32>) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut x0 = Array2::<f32>::zeros((n, d));
        let mut x1 = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for k in 0..d {
                x0[[i, k]] = StandardNormal.sample(&mut rng);
                x1[[i, k]] = StandardNormal.sample(&mut rng);
            }
        }
        (x0, x1)
    }

    fn is_permutation(p: &[usize]) -> bool {
        let n = p.len();
        let mut seen = vec![false; n];
        for &j in p {
            if j >= n || seen[j] {
                return false;
            }
            seen[j] = true;
        }
        true
    }

    #[test]
    fn coupling_is_valid_permutation() {
        let (x0, x1) = make_test_data(16, 4, 42);
        let config = OtCfmConfig::default();
        let coupling = ot_cfm_coupling(&x0.view(), &x1.view(), &config).unwrap();
        assert_eq!(coupling.len(), 16);
        assert!(is_permutation(&coupling));
    }

    #[test]
    fn x_t_at_t0_equals_x0() {
        let (x0, x1) = make_test_data(8, 3, 7);
        let config = OtCfmConfig::default();
        let t = vec![0.0f32; 8];
        let targets = ot_cfm_training_step(&x0.view(), &x1.view(), &t, &config).unwrap();

        for i in 0..8 {
            for k in 0..3 {
                let diff = (targets.x_t[[i, k]] - x0[[i, k]]).abs();
                assert!(diff < 1e-6, "x_t at t=0 should equal x0: diff={diff}");
            }
        }
    }

    #[test]
    fn x_t_at_t1_equals_coupled_x1() {
        let (x0, x1) = make_test_data(8, 3, 13);
        let config = OtCfmConfig::default();
        let t = vec![1.0f32; 8];
        let targets = ot_cfm_training_step(&x0.view(), &x1.view(), &t, &config).unwrap();

        for i in 0..8 {
            let j = targets.coupling[i];
            for k in 0..3 {
                let diff = (targets.x_t[[i, k]] - x1[[j, k]]).abs();
                assert!(
                    diff < 1e-6,
                    "x_t at t=1 should equal x1[coupling[i]]: diff={diff}"
                );
            }
        }
    }

    #[test]
    fn u_t_equals_x1_coupled_minus_x0() {
        let (x0, x1) = make_test_data(8, 3, 99);
        let config = OtCfmConfig::default();
        let t = vec![0.5f32; 8];
        let targets = ot_cfm_training_step(&x0.view(), &x1.view(), &t, &config).unwrap();

        for i in 0..8 {
            let j = targets.coupling[i];
            for k in 0..3 {
                let expected = x1[[j, k]] - x0[[i, k]];
                let diff = (targets.u_t[[i, k]] - expected).abs();
                assert!(
                    diff < 1e-6,
                    "u_t should be x1[coupling[i]] - x0[i]: diff={diff}"
                );
            }
        }
    }

    #[test]
    fn ot_coupling_prefers_nearby_pairs() {
        // Construct data where the identity pairing is optimal:
        // x0 and x1 are nearly identical, with small perturbations.
        let n = 16;
        let d = 4;
        let mut x0 = Array2::<f32>::zeros((n, d));
        let mut x1 = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for k in 0..d {
                let base = (i as f32) * 10.0 + (k as f32) * 0.1;
                x0[[i, k]] = base;
                x1[[i, k]] = base + 0.01; // tiny perturbation
            }
        }

        let config = OtCfmConfig::default();
        let coupling = ot_cfm_coupling(&x0.view(), &x1.view(), &config).unwrap();

        // OT cost with the coupling should be lower than a random permutation.
        let ot_cost: f32 = (0..n)
            .map(|i| {
                let j = coupling[i];
                (0..d)
                    .map(|k| {
                        let diff = x0[[i, k]] - x1[[j, k]];
                        diff * diff
                    })
                    .sum::<f32>()
            })
            .sum();

        // Random permutation: reverse order (far from optimal for this data).
        let random_cost: f32 = (0..n)
            .map(|i| {
                let j = n - 1 - i;
                (0..d)
                    .map(|k| {
                        let diff = x0[[i, k]] - x1[[j, k]];
                        diff * diff
                    })
                    .sum::<f32>()
            })
            .sum();

        assert!(
            ot_cost < random_cost,
            "OT coupling should have lower cost than reverse pairing: ot={ot_cost} random={random_cost}"
        );
    }

    #[test]
    fn shape_mismatch_errors() {
        let x0 = Array2::<f32>::zeros((4, 3));
        let x1_bad_rows = Array2::<f32>::zeros((5, 3));
        let x1_bad_cols = Array2::<f32>::zeros((4, 2));
        let config = OtCfmConfig::default();

        assert!(ot_cfm_coupling(&x0.view(), &x1_bad_rows.view(), &config).is_err());
        assert!(ot_cfm_coupling(&x0.view(), &x1_bad_cols.view(), &config).is_err());

        let t = vec![0.5; 4];
        assert!(ot_cfm_training_step(&x0.view(), &x1_bad_rows.view(), &t, &config).is_err());

        let t_bad = vec![0.5; 3];
        let x1 = Array2::<f32>::zeros((4, 3));
        assert!(ot_cfm_training_step(&x0.view(), &x1.view(), &t_bad, &config).is_err());
    }

    #[test]
    fn t_out_of_range_errors() {
        let x0 = Array2::<f32>::zeros((4, 3));
        let x1 = Array2::<f32>::zeros((4, 3));
        let config = OtCfmConfig::default();

        let t_neg = vec![0.5, -0.1, 0.5, 0.5];
        assert!(ot_cfm_training_step(&x0.view(), &x1.view(), &t_neg, &config).is_err());

        let t_high = vec![0.5, 1.1, 0.5, 0.5];
        assert!(ot_cfm_training_step(&x0.view(), &x1.view(), &t_high, &config).is_err());
    }

    #[test]
    fn invalid_config_errors() {
        let x0 = Array2::<f32>::zeros((4, 3));
        let x1 = Array2::<f32>::zeros((4, 3));

        let bad_reg = OtCfmConfig {
            reg: 0.0,
            ..OtCfmConfig::default()
        };
        assert!(ot_cfm_coupling(&x0.view(), &x1.view(), &bad_reg).is_err());

        let bad_iter = OtCfmConfig {
            max_sinkhorn_iter: 0,
            ..OtCfmConfig::default()
        };
        assert!(ot_cfm_coupling(&x0.view(), &x1.view(), &bad_iter).is_err());

        let bad_tol = OtCfmConfig {
            sinkhorn_tol: -1.0,
            ..OtCfmConfig::default()
        };
        assert!(ot_cfm_coupling(&x0.view(), &x1.view(), &bad_tol).is_err());
    }
}
