//! Euler vs Heun integration on a 2D circular flow.
//!
//! ODE system: dx/dt = -y, dy/dt = x.
//! Exact solution starting from (1, 0): x(t) = cos(t), y(t) = sin(t).
//! The trajectory traces a unit circle, so the radius should stay exactly 1.
//!
//! Euler (1st order) spirals outward; Heun (2nd order) tracks the circle much better.
//! This example prints the trajectory at each step and the final radius error.

use flowmatch::ode::{integrate_fixed, OdeMethod};
use ndarray::Array1;

fn main() {
    let x0 = Array1::from_vec(vec![1.0f32, 0.0]);
    let total_t = 2.0 * std::f32::consts::PI; // one full revolution
    let step_counts = [10, 20, 50, 100, 200];

    println!("=== Euler vs Heun on circular ODE: dx/dt = -y, dy/dt = x ===");
    println!("Exact solution: (cos(t), sin(t)), radius = 1.0 for all t.");
    println!();

    // Velocity field: v(x, t) = (-y, x).
    let velocity = |x: &ndarray::ArrayView1<f32>, _t: f32| -> flowmatch::Result<Array1<f32>> {
        Ok(Array1::from_vec(vec![-x[1], x[0]]))
    };

    println!(
        "{:>8}  {:>14}  {:>14}  {:>14}  {:>14}",
        "steps", "euler_r", "euler_err", "heun_r", "heun_err"
    );
    println!("{}", "-".repeat(72));

    for &steps in &step_counts {
        let dt = total_t / (steps as f32);

        let euler = integrate_fixed(OdeMethod::Euler, &x0, 0.0, dt, steps, velocity).unwrap();
        let heun = integrate_fixed(OdeMethod::Heun, &x0, 0.0, dt, steps, velocity).unwrap();

        let r_euler = (euler[0] * euler[0] + euler[1] * euler[1]).sqrt();
        let r_heun = (heun[0] * heun[0] + heun[1] * heun[1]).sqrt();

        let err_euler = (r_euler - 1.0).abs();
        let err_heun = (r_heun - 1.0).abs();

        println!(
            "{:>8}  {:>14.8}  {:>14.8}  {:>14.8}  {:>14.8}",
            steps, r_euler, err_euler, r_heun, err_heun
        );
    }

    println!();

    // Show a trajectory trace with 20 steps.
    let steps = 20;
    let dt = total_t / (steps as f32);
    println!("--- Trajectory trace ({steps} steps over one full revolution) ---");
    println!();
    println!(
        "{:>4}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "step", "euler_x", "euler_y", "euler_r", "heun_x", "heun_y", "heun_r"
    );

    let mut xe = x0.clone();
    let mut xh = x0.clone();

    for i in 0..=steps {
        let re = (xe[0] * xe[0] + xe[1] * xe[1]).sqrt();
        let rh = (xh[0] * xh[0] + xh[1] * xh[1]).sqrt();
        println!(
            "{:>4}  {:>10.5}  {:>10.5}  {:>10.5}  {:>10.5}  {:>10.5}  {:>10.5}",
            i, xe[0], xe[1], re, xh[0], xh[1], rh
        );

        if i < steps {
            // Euler step.
            let ve = velocity(&xe.view(), 0.0).unwrap();
            let mut xe_new = xe.clone();
            for k in 0..2 {
                xe_new[k] += dt * ve[k];
            }
            xe = xe_new;

            // Heun step.
            let v0 = velocity(&xh.view(), 0.0).unwrap();
            let mut xh_pred = xh.clone();
            for k in 0..2 {
                xh_pred[k] += dt * v0[k];
            }
            let v1 = velocity(&xh_pred.view(), 0.0).unwrap();
            let mut xh_new = xh.clone();
            for k in 0..2 {
                xh_new[k] += 0.5 * dt * (v0[k] + v1[k]);
            }
            xh = xh_new;
        }
    }

    println!();
    println!("Observation: Euler's radius grows each step (energy gain),");
    println!("while Heun's radius stays near 1.0 (symplectic-like behavior).");
}
