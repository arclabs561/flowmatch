//! Minimal ODE integrators for sampling flow models.
//!
//! This crate uses ODE sampling of the form:
//! \[
//! \frac{dx}{dt} = v_\theta(x,t;\cdot)
//! \]
//!
//! We keep this module tiny and deterministic: no adaptive stepping, no hidden tolerances.

use ndarray::{Array1, ArrayView1};

/// Fixed-step ODE method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeMethod {
    /// Explicit Euler (1st order).
    Euler,
    /// Heun / explicit trapezoid / RK2 (2nd order).
    Heun,
}

/// Integrate an ODE forward with fixed steps on a vector state.
///
/// - `x0`: initial state
/// - `t0`: initial time
/// - `dt`: step size
/// - `steps`: number of steps (must be >= 1)
/// - `f`: returns velocity \(v(x,t)\)
pub fn integrate_fixed(
    method: OdeMethod,
    x0: &Array1<f32>,
    t0: f32,
    dt: f32,
    steps: usize,
    mut f: impl FnMut(&ArrayView1<f32>, f32) -> Array1<f32>,
) -> Array1<f32> {
    assert!(steps >= 1);
    assert!(dt.is_finite());

    let mut x = x0.clone();
    let mut t = t0;

    match method {
        OdeMethod::Euler => {
            for _ in 0..steps {
                let v = f(&x.view(), t);
                // x += dt * v
                for i in 0..x.len() {
                    x[i] += dt * v[i];
                }
                t += dt;
            }
        }
        OdeMethod::Heun => {
            for _ in 0..steps {
                let v0 = f(&x.view(), t);

                // predictor
                let mut x_pred = x.clone();
                for i in 0..x.len() {
                    x_pred[i] += dt * v0[i];
                }

                // corrector
                let v1 = f(&x_pred.view(), t + dt);
                for i in 0..x.len() {
                    x[i] += 0.5 * dt * (v0[i] + v1[i]);
                }

                t += dt;
            }
        }
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn heun_is_more_accurate_than_euler_on_dx_dt_eq_minus_x() {
        // ODE: dx/dt = -x, x(0)=1, exact x(1)=e^-1.
        let x0 = Array1::from_vec(vec![1.0f32]);
        let exact = (-1.0f32).exp();

        let steps = 20usize;
        let dt = 1.0f32 / (steps as f32);

        let euler = integrate_fixed(OdeMethod::Euler, &x0, 0.0, dt, steps, |x, _t| {
            Array1::from_vec(vec![-x[0]])
        });
        let heun = integrate_fixed(OdeMethod::Heun, &x0, 0.0, dt, steps, |x, _t| {
            Array1::from_vec(vec![-x[0]])
        });

        let err_euler = (euler[0] - exact).abs();
        let err_heun = (heun[0] - exact).abs();

        assert!(
            err_heun < err_euler,
            "expected Heun to be more accurate: err_heun={err_heun} err_euler={err_euler}"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_constant_field_is_exact_for_euler_and_heun(
            len in 1usize..16,
            steps in 1usize..200,
            dt in 1e-3f32..1.0f32,
            t0 in -2.0f32..2.0f32,
            x0 in prop::collection::vec(-10.0f32..10.0f32, 16),
            c in prop::collection::vec(-10.0f32..10.0f32, 16),
        ) {
            let x0 = Array1::from_vec(x0[..len].to_vec());
            let c = Array1::from_vec(c[..len].to_vec());

            let expected = {
                let mut out = x0.clone();
                let scale = dt * (steps as f32);
                for i in 0..len {
                    out[i] += scale * c[i];
                }
                out
            };

            let euler = integrate_fixed(OdeMethod::Euler, &x0, t0, dt, steps, |_x, _t| c.clone());
            let heun = integrate_fixed(OdeMethod::Heun, &x0, t0, dt, steps, |_x, _t| c.clone());

            for i in 0..len {
                // Constant fields are "exact" in the method sense, but floating addition accumulates
                // rounding error over many steps. Use a small absolute + relative tolerance.
                // Worst-case float accumulation can be noticeable for large `steps` and `dt`.
                // This is still a strong invariant: error should be small compared to the total drift.
                let tol = 2e-2 + 1e-6 * expected[i].abs();
                prop_assert!((euler[i] - expected[i]).abs() <= tol, "euler mismatch at {i}");
                prop_assert!((heun[i] - expected[i]).abs() <= tol, "heun mismatch at {i}");
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_error_decreases_with_more_steps_for_dx_dt_eq_minus_x(
            steps in 5usize..80,
        ) {
            let x0 = Array1::from_vec(vec![1.0f32]);
            let exact = (-1.0f32).exp();

            let dt1 = 1.0f32 / (steps as f32);
            let dt2 = 1.0f32 / ((2 * steps) as f32);

            let e1 = integrate_fixed(OdeMethod::Euler, &x0, 0.0, dt1, steps, |x, _t| {
                Array1::from_vec(vec![-x[0]])
            });
            let e2 = integrate_fixed(OdeMethod::Euler, &x0, 0.0, dt2, 2 * steps, |x, _t| {
                Array1::from_vec(vec![-x[0]])
            });

            let h1 = integrate_fixed(OdeMethod::Heun, &x0, 0.0, dt1, steps, |x, _t| {
                Array1::from_vec(vec![-x[0]])
            });
            let h2 = integrate_fixed(OdeMethod::Heun, &x0, 0.0, dt2, 2 * steps, |x, _t| {
                Array1::from_vec(vec![-x[0]])
            });

            let err_e1 = (e1[0] - exact).abs();
            let err_e2 = (e2[0] - exact).abs();
            let err_h1 = (h1[0] - exact).abs();
            let err_h2 = (h2[0] - exact).abs();

            // With smaller dt, error should not get worse (allow tiny numerical wiggle).
            prop_assert!(err_e2 <= err_e1 + 1e-6, "euler error did not decrease: {err_e1} -> {err_e2}");
            prop_assert!(err_h2 <= err_h1 + 1e-6, "heun error did not decrease: {err_h1} -> {err_h2}");

            // Heun should generally be at least as accurate as Euler at the same dt.
            prop_assert!(err_h1 <= err_e1 + 1e-6, "expected Heun <= Euler at steps={steps}");
        }
    }
}
