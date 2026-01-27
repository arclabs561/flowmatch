//! Simple vector-field parameterizations for flow matching.
//!
//! These are intentionally boring baselines: enough structure to test the training loop,
//! without importing a full ML framework.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// A linear vector field that is conditioned on a discrete target point `y`:
///
/// \[
/// v_\theta(x, t; y) = W \cdot [x; y; t; 1],
/// \]
///
/// where `W` is a `d × (2d+2)` matrix.
#[derive(Debug, Clone)]
pub struct LinearCondField {
    /// Parameters `W` with shape `(d, 2d+2)`.
    pub w: Array2<f32>,
}

impl LinearCondField {
    pub fn new_zeros(d: usize) -> Self {
        Self {
            w: Array2::zeros((d, 2 * d + 2)),
        }
    }

    pub fn d(&self) -> usize {
        self.w.nrows()
    }

    /// Evaluate the field at `(x, t)` conditioned on a fixed `y`.
    pub fn eval(&self, x: &ArrayView1<f32>, t: f32, y: &ArrayView1<f32>) -> Array1<f32> {
        let d = self.d();
        debug_assert_eq!(x.len(), d);
        debug_assert_eq!(y.len(), d);

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
        out
    }

    /// One SGD step on mean-squared error:
    ///
    /// \[
    /// L = \tfrac12 \|v_\theta(x,t;y) - u\|_2^2.
    /// \]
    pub fn sgd_step(
        &mut self,
        x: &ArrayView1<f32>,
        t: f32,
        y: &ArrayView1<f32>,
        u: &ArrayView1<f32>,
        lr: f32,
    ) {
        let d = self.d();
        debug_assert_eq!(x.len(), d);
        debug_assert_eq!(y.len(), d);
        debug_assert_eq!(u.len(), d);

        // pred and residual
        let pred = self.eval(x, t, y);
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
    }

    /// Mean squared error averaged over a batch.
    pub fn mse_batch(
        &self,
        xs: &ArrayView2<f32>,
        ts: &[f32],
        ys: &ArrayView2<f32>,
        us: &ArrayView2<f32>,
    ) -> f32 {
        let n = xs.nrows();
        let d = xs.ncols();
        debug_assert_eq!(ys.nrows(), n);
        debug_assert_eq!(ys.ncols(), d);
        debug_assert_eq!(us.nrows(), n);
        debug_assert_eq!(us.ncols(), d);
        debug_assert_eq!(ts.len(), n);

        let mut s: f64 = 0.0;
        for i in 0..n {
            let pred = self.eval(&xs.row(i), ts[i], &ys.row(i));
            for k in 0..d {
                let r = (pred[k] - us[[i, k]]) as f64;
                s += r * r;
            }
        }
        (s / (n as f64 * d as f64)) as f32
    }
}
