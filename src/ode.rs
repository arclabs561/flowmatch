//! Minimal ODE integrators for sampling flow models.
//!
//! This crate uses ODE sampling of the form:
//! \[
//! \frac{dx}{dt} = v_\theta(x,t;\cdot)
//! \]
//!
//! We keep this module tiny and deterministic: no adaptive stepping, no hidden tolerances.
//!
//! # Discretization error analysis
//!
//! Guan et al., "Total Variation Rates for Riemannian Flow Matching" (2026)
//! provides the first nonasymptotic TV convergence analysis for Riemannian
//! flow matching sampling. It quantifies how Euler discretization error
//! propagates to sample quality, giving concrete bounds on the number of
//! integration steps needed for a target TV tolerance.

use ndarray::{Array1, ArrayView1};

/// Fixed-step ODE method.
///
/// In flow matching, learned velocity fields are generally not constant along
/// trajectories -- paths curve even after rectification. Heun (2nd order)
/// compensates for this curvature and typically needs fewer total function
/// evaluations than Euler for equivalent endpoint accuracy, despite costing
/// 2 evaluations per step.
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
/// - `f`: returns velocity `v(x,t)`
pub fn integrate_fixed(
    method: OdeMethod,
    x0: &Array1<f32>,
    t0: f32,
    dt: f32,
    steps: usize,
    mut f: impl FnMut(&ArrayView1<f32>, f32) -> crate::Result<Array1<f32>>,
) -> crate::Result<Array1<f32>> {
    if steps < 1 {
        return Err(crate::Error::Domain("steps must be >= 1"));
    }
    if !dt.is_finite() {
        return Err(crate::Error::Domain("dt must be finite"));
    }

    let mut x = x0.clone();
    let mut t = t0;

    match method {
        OdeMethod::Euler => {
            for _ in 0..steps {
                let v = f(&x.view(), t)?;
                // x += dt * v
                for i in 0..x.len() {
                    x[i] += dt * v[i];
                }
                t += dt;
            }
        }
        OdeMethod::Heun => {
            for _ in 0..steps {
                let v0 = f(&x.view(), t)?;

                // predictor
                let mut x_pred = x.clone();
                for i in 0..x.len() {
                    x_pred[i] += dt * v0[i];
                }

                // corrector
                let v1 = f(&x_pred.view(), t + dt)?;
                for i in 0..x.len() {
                    x[i] += 0.5 * dt * (v0[i] + v1[i]);
                }

                t += dt;
            }
        }
    }

    Ok(x)
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
            Ok(Array1::from_vec(vec![-x[0]]))
        })
        .unwrap();
        let heun = integrate_fixed(OdeMethod::Heun, &x0, 0.0, dt, steps, |x, _t| {
            Ok(Array1::from_vec(vec![-x[0]]))
        })
        .unwrap();

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

            let euler = integrate_fixed(OdeMethod::Euler, &x0, t0, dt, steps, |_x, _t| Ok(c.clone())).unwrap();
            let heun = integrate_fixed(OdeMethod::Heun, &x0, t0, dt, steps, |_x, _t| Ok(c.clone())).unwrap();

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
                Ok(Array1::from_vec(vec![-x[0]]))
            }).unwrap();
            let e2 = integrate_fixed(OdeMethod::Euler, &x0, 0.0, dt2, 2 * steps, |x, _t| {
                Ok(Array1::from_vec(vec![-x[0]]))
            }).unwrap();

            let h1 = integrate_fixed(OdeMethod::Heun, &x0, 0.0, dt1, steps, |x, _t| {
                Ok(Array1::from_vec(vec![-x[0]]))
            }).unwrap();
            let h2 = integrate_fixed(OdeMethod::Heun, &x0, 0.0, dt2, 2 * steps, |x, _t| {
                Ok(Array1::from_vec(vec![-x[0]]))
            }).unwrap();

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

    #[test]
    fn euler_on_exponential_decay_converges_to_exact() {
        // ODE: dx/dt = -x, x(0) = 1. Exact: x(1) = e^{-1} ~ 0.3679.
        // With 1000 steps, Euler should be within 1e-3 of exact.
        let x0 = Array1::from_vec(vec![1.0f32]);
        let exact = (-1.0f32).exp();
        let steps = 1000usize;
        let dt = 1.0f32 / (steps as f32);

        let result = integrate_fixed(OdeMethod::Euler, &x0, 0.0, dt, steps, |x, _t| {
            Ok(Array1::from_vec(vec![-x[0]]))
        })
        .unwrap();

        let err = (result[0] - exact).abs();
        assert!(
            err < 1e-3,
            "Euler with 1000 steps should be within 1e-3 of exact: got {}, exact {}, err {}",
            result[0],
            exact,
            err
        );
    }

    #[test]
    fn heun_on_exponential_decay_is_very_accurate() {
        // Heun is 2nd order, so with 100 steps on dx/dt=-x it should be extremely accurate.
        let x0 = Array1::from_vec(vec![1.0f32]);
        let exact = (-1.0f32).exp();
        let steps = 100usize;
        let dt = 1.0f32 / (steps as f32);

        let result = integrate_fixed(OdeMethod::Heun, &x0, 0.0, dt, steps, |x, _t| {
            Ok(Array1::from_vec(vec![-x[0]]))
        })
        .unwrap();

        let err = (result[0] - exact).abs();
        assert!(
            err < 1e-5,
            "Heun with 100 steps should be within 1e-5 of exact: got {}, exact {}, err {}",
            result[0],
            exact,
            err
        );
    }

    #[test]
    fn euler_and_heun_on_2d_rotation_preserve_radius() {
        // ODE: dx/dt = -y, dy/dt = x. Exact solution traces a circle of radius |x0|.
        // Euler spirals outward; Heun should preserve radius much better.
        let x0 = Array1::from_vec(vec![1.0f32, 0.0]);
        let r0 = 1.0f32;
        let steps = 200usize;
        let total_t = std::f32::consts::PI; // half rotation
        let dt = total_t / (steps as f32);

        let euler = integrate_fixed(OdeMethod::Euler, &x0, 0.0, dt, steps, |x, _t| {
            Ok(Array1::from_vec(vec![-x[1], x[0]]))
        })
        .unwrap();
        let heun = integrate_fixed(OdeMethod::Heun, &x0, 0.0, dt, steps, |x, _t| {
            Ok(Array1::from_vec(vec![-x[1], x[0]]))
        })
        .unwrap();

        let r_euler = (euler[0] * euler[0] + euler[1] * euler[1]).sqrt();
        let r_heun = (heun[0] * heun[0] + heun[1] * heun[1]).sqrt();

        let err_euler = (r_euler - r0).abs();
        let err_heun = (r_heun - r0).abs();

        // Heun preserves radius much better than Euler on this rotation.
        assert!(
            err_heun < err_euler,
            "Heun should preserve radius better: err_heun={err_heun} err_euler={err_euler}"
        );
        // Heun error should be small in absolute terms.
        assert!(
            err_heun < 0.01,
            "Heun radius error should be < 0.01, got {err_heun}"
        );
    }

    #[test]
    fn single_step_euler_matches_manual() {
        // One Euler step: x1 = x0 + dt * f(x0, t0).
        let x0 = Array1::from_vec(vec![2.0f32, 3.0]);
        let dt = 0.1f32;
        let result = integrate_fixed(OdeMethod::Euler, &x0, 0.0, dt, 1, |x, _t| {
            // f(x) = [x[1], -x[0]]
            Ok(Array1::from_vec(vec![x[1], -x[0]]))
        })
        .unwrap();
        // x1 = [2.0 + 0.1*3.0, 3.0 + 0.1*(-2.0)] = [2.3, 2.8]
        assert!((result[0] - 2.3).abs() < 1e-6);
        assert!((result[1] - 2.8).abs() < 1e-6);
    }
}
