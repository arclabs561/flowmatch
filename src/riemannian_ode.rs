//! Riemannian ODE sampling utilities.
//!
//! This is the manifold analogue of `ode::integrate_fixed`, where states live on a manifold and
//! velocities live in tangent spaces. The key difference is that higher-order methods (like Heun)
//! must **parallel transport** tangent vectors to a common tangent space before combining them.

use ndarray::{Array1, ArrayView1};
use skel::Manifold;

use crate::ode::OdeMethod;

/// Integrate a Riemannian ODE forward with fixed steps on a manifold.
///
/// ODE: \( \dot{x}(t) = v(x(t), t) \) where \(v(x,t) \in T_x M\).
///
/// - `method`: `Euler` or `Heun`.
/// - `x0`: initial point on the manifold.
/// - `t0`: initial time.
/// - `dt`: step size.
/// - `steps`: number of steps (must be >= 1).
/// - `f`: returns a tangent vector in the current tangent space \(T_x M\).
pub fn integrate_fixed_manifold<M>(
    method: OdeMethod,
    manifold: &M,
    x0: &Array1<f64>,
    t0: f64,
    dt: f64,
    steps: usize,
    mut f: impl FnMut(&ArrayView1<f64>, f64) -> Array1<f64>,
) -> Array1<f64>
where
    M: Manifold,
{
    assert!(steps >= 1);
    assert!(dt.is_finite());

    let mut x = x0.clone();
    let mut t = t0;

    match method {
        OdeMethod::Euler => {
            for _ in 0..steps {
                let v = f(&x.view(), t);
                let step = v.mapv(|u| u * dt);
                x = manifold.exp_map(&x.view(), &step.view());
                t += dt;
            }
        }
        OdeMethod::Heun => {
            for _ in 0..steps {
                let v0 = f(&x.view(), t);

                // predictor (Euler)
                let step0 = v0.mapv(|u| u * dt);
                let x_pred = manifold.exp_map(&x.view(), &step0.view());

                // corrector velocity lives in T_{x_pred} M
                let v1 = f(&x_pred.view(), t + dt);

                // Bring v1 back to T_x M so we can average in one tangent space.
                let v1_at_x = manifold.parallel_transport(&x_pred.view(), &x.view(), &v1.view());

                let v_avg = (&v0 + &v1_at_x).mapv(|u| 0.5 * u);
                let step = v_avg.mapv(|u| u * dt);
                x = manifold.exp_map(&x.view(), &step.view());

                t += dt;
            }
        }
    }

    x
}

#[cfg(all(test, feature = "hyp-riemannian"))]
mod tests {
    use super::*;
    use hyp::PoincareBall;
    use proptest::prelude::*;

    fn poincare_point() -> impl Strategy<Value = Array1<f64>> {
        prop::collection::vec(-0.6f64..0.6f64, 2).prop_map(|v| {
            let x = Array1::from_vec(v);
            let norm = x.dot(&x).sqrt();
            if norm > 0.75 {
                x * (0.75 / norm)
            } else {
                x
            }
        })
    }

    fn small_vec2() -> impl Strategy<Value = Array1<f64>> {
        prop::collection::vec(-0.2f64..0.2f64, 2).prop_map(Array1::from_vec)
    }

    #[test]
    fn heun_tracks_geodesic_better_than_euler_smoke() {
        let m = PoincareBall::<f64>::new(1.0);
        let x0 = Array1::from_vec(vec![0.05, -0.02]);
        let v0 = Array1::from_vec(vec![0.12, 0.04]);

        let exact = m.exp_map(&x0.view(), &v0.view());

        let steps = 64usize;
        let dt = 1.0f64 / (steps as f64);

        let euler = integrate_fixed_manifold(OdeMethod::Euler, &m, &x0, 0.0, dt, steps, |x, _t| {
            m.parallel_transport(&x0.view(), x, &v0.view())
        });
        let heun = integrate_fixed_manifold(OdeMethod::Heun, &m, &x0, 0.0, dt, steps, |x, _t| {
            m.parallel_transport(&x0.view(), x, &v0.view())
        });

        let err_e = (&euler - &exact).dot(&(&euler - &exact)).sqrt();
        let err_h = (&heun - &exact).dot(&(&heun - &exact)).sqrt();

        assert!(
            err_h <= err_e + 1e-6,
            "expected Heun <= Euler: err_heun={err_h} err_euler={err_e}"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]

        #[test]
        fn prop_error_decreases_with_more_steps_on_geodesic_field(
            x0 in poincare_point(),
            v0 in small_vec2(),
            steps in 10usize..80,
        ) {
            let m = PoincareBall::<f64>::new(1.0);
            let exact = m.exp_map(&x0.view(), &v0.view());

            let dt1 = 1.0f64 / (steps as f64);
            let dt2 = 1.0f64 / ((2 * steps) as f64);

            let e1 = integrate_fixed_manifold(OdeMethod::Euler, &m, &x0, 0.0, dt1, steps, |x, _t| {
                m.parallel_transport(&x0.view(), x, &v0.view())
            });
            let e2 = integrate_fixed_manifold(OdeMethod::Euler, &m, &x0, 0.0, dt2, 2 * steps, |x, _t| {
                m.parallel_transport(&x0.view(), x, &v0.view())
            });

            let h1 = integrate_fixed_manifold(OdeMethod::Heun, &m, &x0, 0.0, dt1, steps, |x, _t| {
                m.parallel_transport(&x0.view(), x, &v0.view())
            });
            let h2 = integrate_fixed_manifold(OdeMethod::Heun, &m, &x0, 0.0, dt2, 2 * steps, |x, _t| {
                m.parallel_transport(&x0.view(), x, &v0.view())
            });

            let err_e1 = (&e1 - &exact).dot(&(&e1 - &exact)).sqrt();
            let err_e2 = (&e2 - &exact).dot(&(&e2 - &exact)).sqrt();
            let err_h1 = (&h1 - &exact).dot(&(&h1 - &exact)).sqrt();
            let err_h2 = (&h2 - &exact).dot(&(&h2 - &exact)).sqrt();

            // Euler is only first-order and can show small non-monotone wiggles under floating error
            // (and here, also under numerical parallel transport). Heun should be the reliably
            // convergent method in this setting.
            prop_assert!(err_h2 <= err_h1 + 2e-6, "heun error did not decrease: {err_h1} -> {err_h2}");

            // Sanity: refinement should not catastrophically worsen Euler.
            prop_assert!(err_e2 <= err_e1 + 2e-5, "euler got much worse: {err_e1} -> {err_e2}");
        }
    }
}
