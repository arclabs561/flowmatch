//! Simple vector-field parameterizations for flow matching.
//!
//! These are intentionally boring baselines: enough structure to test the training loop,
//! without importing a full ML framework.
//!
//! # References
//!
//! - Lipman et al., "Flow Matching for Generative Modeling" (2022) --
//!   foundational paper establishing conditional flow matching with linear
//!   interpolation paths between noise and data.
//! - Liu, "Rectified Flow" (2022) -- the reflow procedure that iteratively
//!   straightens learned flow paths, reducing ODE integration error.
//! - Roy et al., "2-Rectifications are Enough" (2024) -- proves that two
//!   rounds of reflow suffice for theoretical convergence guarantees on
//!   path straightness.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// A linear vector field that is conditioned on a discrete target point `y`:
///
/// $$
/// v_\theta(x, t; y) = W \cdot [x; y; t; 1],
/// $$
///
/// where `W` is a `d × (2d+2)` matrix.
#[derive(Debug, Clone)]
pub struct LinearCondField {
    /// Parameters `W` with shape `(d, 2d+2)`.
    pub w: Array2<f32>,
}

impl LinearCondField {
    /// Create a zero-initialized field for dimension `d`.
    pub fn new_zeros(d: usize) -> Self {
        Self {
            w: Array2::zeros((d, 2 * d + 2)),
        }
    }

    /// Output dimension of the field.
    pub fn d(&self) -> usize {
        self.w.nrows()
    }

    /// Evaluate the field at `(x, t)` conditioned on a fixed `y`.
    pub fn eval(
        &self,
        x: &ArrayView1<f32>,
        t: f32,
        y: &ArrayView1<f32>,
    ) -> crate::Result<Array1<f32>> {
        let d = self.d();
        if x.len() != d || y.len() != d {
            return Err(crate::Error::Shape("x and y must have length d"));
        }

        // features = [x (d), y (d), t, 1]
        let mut out = Array1::<f32>::zeros(d);
        for i in 0..d {
            let mut s = 0.0f32;
            // x part
            for k in 0..d {
                s += self.w[[i, k]] * x[k];
            }
            // y part
            for k in 0..d {
                s += self.w[[i, d + k]] * y[k];
            }
            // t and bias
            s += self.w[[i, 2 * d]] * t;
            s += self.w[[i, 2 * d + 1]] * 1.0;
            out[i] = s;
        }
        Ok(out)
    }

    /// One SGD step on mean-squared error:
    ///
    /// $$
    /// L = \tfrac{1}{2} \|v_\theta(x,t;y) - u\|_2^2.
    /// $$
    pub fn sgd_step(
        &mut self,
        x: &ArrayView1<f32>,
        t: f32,
        y: &ArrayView1<f32>,
        u: &ArrayView1<f32>,
        lr: f32,
    ) -> crate::Result<()> {
        let d = self.d();
        if x.len() != d || y.len() != d || u.len() != d {
            return Err(crate::Error::Shape("x, y, and u must have length d"));
        }

        // pred and residual
        let pred = self.eval(x, t, y)?;
        let mut r = Array1::<f32>::zeros(d);
        for i in 0..d {
            r[i] = pred[i] - u[i];
        }

        // Gradient: dW[i, feat] = r[i] * feat.
        // Apply SGD: W -= lr * grad.
        for i in 0..d {
            let ri = r[i];
            // x features
            for k in 0..d {
                self.w[[i, k]] -= lr * ri * x[k];
            }
            // y features
            for k in 0..d {
                self.w[[i, d + k]] -= lr * ri * y[k];
            }
            // t + bias
            self.w[[i, 2 * d]] -= lr * ri * t;
            self.w[[i, 2 * d + 1]] -= lr * ri * 1.0;
        }
        Ok(())
    }

    /// Mean squared error averaged over a batch.
    pub fn mse_batch(
        &self,
        xs: &ArrayView2<f32>,
        ts: &[f32],
        ys: &ArrayView2<f32>,
        us: &ArrayView2<f32>,
    ) -> crate::Result<f32> {
        let n = xs.nrows();
        let d = xs.ncols();
        if ys.nrows() != n || ys.ncols() != d || us.nrows() != n || us.ncols() != d || ts.len() != n
        {
            return Err(crate::Error::Shape("batch dimensions must agree"));
        }

        let mut s: f64 = 0.0;
        for i in 0..n {
            let pred = self.eval(&xs.row(i), ts[i], &ys.row(i))?;
            for k in 0..d {
                let r = (pred[k] - us[[i, k]]) as f64;
                s += r * r;
            }
        }
        Ok((s / (n as f64 * d as f64)) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_field_evals_to_zero() {
        let f = LinearCondField::new_zeros(3);
        let x = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0f32, 5.0, 6.0]);
        let out = f.eval(&x.view(), 0.5, &y.view()).unwrap();
        for i in 0..3 {
            assert_eq!(
                out[i], 0.0,
                "zero field must produce zero output at dim {i}"
            );
        }
    }

    #[test]
    fn eval_matches_manual_1d() {
        // d=1 => W is (1, 4): [w_x, w_y, w_t, w_bias]
        let mut f = LinearCondField::new_zeros(1);
        f.w[[0, 0]] = 2.0; // w_x
        f.w[[0, 1]] = 3.0; // w_y
        f.w[[0, 2]] = 4.0; // w_t
        f.w[[0, 3]] = 5.0; // w_bias

        let x = Array1::from_vec(vec![1.0f32]);
        let y = Array1::from_vec(vec![2.0f32]);
        let t = 0.5f32;

        let out = f.eval(&x.view(), t, &y.view()).unwrap();
        // out = 2*1 + 3*2 + 4*0.5 + 5*1 = 2 + 6 + 2 + 5 = 15
        assert!((out[0] - 15.0).abs() < 1e-6, "got {}", out[0]);
    }

    #[test]
    fn sgd_step_reduces_loss_on_constant_target() {
        // Train on a single example repeatedly; loss should decrease.
        let mut f = LinearCondField::new_zeros(2);
        let x = Array1::from_vec(vec![1.0f32, 0.0]);
        let y = Array1::from_vec(vec![0.0f32, 1.0]);
        let u = Array1::from_vec(vec![3.0f32, -2.0]); // target velocity
        let t = 0.5f32;
        let lr = 0.01f32;

        let pred_before = f.eval(&x.view(), t, &y.view()).unwrap();
        let loss_before: f32 = pred_before
            .iter()
            .zip(u.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();

        for _ in 0..100 {
            f.sgd_step(&x.view(), t, &y.view(), &u.view(), lr).unwrap();
        }

        let pred_after = f.eval(&x.view(), t, &y.view()).unwrap();
        let loss_after: f32 = pred_after
            .iter()
            .zip(u.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();

        assert!(
            loss_after < loss_before * 0.01,
            "SGD should reduce loss substantially: before={loss_before} after={loss_after}"
        );
    }

    #[test]
    fn mse_batch_zero_field_equals_target_norm() {
        // Zero field => pred = 0, so MSE = mean(||u||^2 / d).
        let f = LinearCondField::new_zeros(2);
        let xs = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let ys = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let us = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 1.0, 2.0]).unwrap();
        let ts = vec![0.3f32, 0.7];

        let mse = f
            .mse_batch(&xs.view(), &ts, &ys.view(), &us.view())
            .unwrap();
        // Expected: ( (9+16) + (1+4) ) / (2*2) = 30/4 = 7.5
        assert!((mse - 7.5).abs() < 1e-5, "expected 7.5, got {mse}");
    }
}
