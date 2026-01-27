//! Motivating example: manifold ODE integration on the Poincaré ball.
//!
//! We integrate a “geodesic velocity field” starting at `x0` with initial tangent `v0`. The true
//! solution at t=1 is `exp_{x0}(v0)`. Heun requires parallel transport to average tangent vectors.

use flowmatch::ode::OdeMethod;
use flowmatch::riemannian_ode::integrate_fixed_manifold;
use hyp::PoincareBall;
use ndarray::Array1;
use skel::Manifold;

fn main() {
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

    println!("x0      = {x0:?}");
    println!("v0      = {v0:?}");
    println!("exact   = {exact:?}");
    println!("euler   = {euler:?} (err {err_e:.6e})");
    println!("heun    = {heun:?} (err {err_h:.6e})");
}
