//! Verify that Euler achieves O(h) and Heun achieves O(h^2) convergence.
//!
//! Uses dx/dt = -x with exact solution x(T) = x(0) * exp(-T).
//! Runs at multiple step counts and checks the error decay rate.

use flowmatch::ode::{integrate_fixed, OdeMethod};
use ndarray::Array1;

/// Solve dx/dt = -x from t=0 to t=1, return absolute error vs exact.
fn error_at_steps(method: OdeMethod, steps: usize) -> f64 {
    let x0 = Array1::from_vec(vec![1.0f32]);
    let exact = (-1.0f32).exp();
    let dt = 1.0f32 / (steps as f32);

    let result = integrate_fixed(method, &x0, 0.0, dt, steps, |x, _t| {
        Ok(Array1::from_vec(vec![-x[0]]))
    })
    .unwrap();

    (result[0] - exact).abs() as f64
}

/// Estimate convergence order from errors at doubling step counts.
///
/// Returns the average of log2(err[i] / err[i+1]) across consecutive pairs.
fn estimate_order(method: OdeMethod, step_counts: &[usize]) -> f64 {
    let errors: Vec<f64> = step_counts
        .iter()
        .map(|&s| error_at_steps(method, s))
        .collect();

    let mut ratios = Vec::new();
    for i in 0..errors.len() - 1 {
        if errors[i + 1] > 1e-12 {
            // When steps double, h halves. Order p means err(h/2) / err(h) ~ 2^{-p}.
            let ratio = (errors[i] / errors[i + 1]).ln() / 2.0f64.ln();
            ratios.push(ratio);
        }
    }

    ratios.iter().sum::<f64>() / ratios.len() as f64
}

#[test]
fn euler_achieves_first_order_convergence() {
    // Doubling step counts: 10, 20, 40, 80, 160.
    let steps = vec![10, 20, 40, 80, 160];
    let order = estimate_order(OdeMethod::Euler, &steps);

    assert!(
        (0.85..1.15).contains(&order),
        "Euler should be ~O(h^1), estimated order = {order:.3}"
    );
}

#[test]
fn heun_achieves_second_order_convergence() {
    // Doubling step counts: 10, 20, 40, 80, 160.
    let steps = vec![10, 20, 40, 80, 160];
    let order = estimate_order(OdeMethod::Heun, &steps);

    assert!(
        (1.85..2.15).contains(&order),
        "Heun should be ~O(h^2), estimated order = {order:.3}"
    );
}

/// Same test on a second ODE to confirm rates are not problem-specific.
///
/// dx/dt = cos(t), x(0) = 0. Exact: x(T) = sin(T).
fn error_cosine_at_steps(method: OdeMethod, steps: usize) -> f64 {
    let x0 = Array1::from_vec(vec![0.0f32]);
    let t_final = 1.0f32;
    let exact = t_final.sin();
    let dt = t_final / (steps as f32);

    let result = integrate_fixed(method, &x0, 0.0, dt, steps, |_x, t| {
        Ok(Array1::from_vec(vec![t.cos()]))
    })
    .unwrap();

    (result[0] - exact).abs() as f64
}

#[test]
fn euler_first_order_on_cosine_ode() {
    let steps = vec![20, 40, 80, 160, 320];
    let errors: Vec<f64> = steps
        .iter()
        .map(|&s| error_cosine_at_steps(OdeMethod::Euler, s))
        .collect();

    let mut ratios = Vec::new();
    for i in 0..errors.len() - 1 {
        if errors[i + 1] > 1e-12 {
            ratios.push((errors[i] / errors[i + 1]).ln() / 2.0f64.ln());
        }
    }
    let order = ratios.iter().sum::<f64>() / ratios.len() as f64;

    assert!(
        (0.85..1.15).contains(&order),
        "Euler should be ~O(h^1) on cosine ODE, estimated order = {order:.3}"
    );
}

#[test]
fn heun_second_order_on_cosine_ode() {
    // Keep step counts low to stay in the truncation-dominated regime (f32).
    let steps = vec![5, 10, 20, 40, 80];
    let errors: Vec<f64> = steps
        .iter()
        .map(|&s| error_cosine_at_steps(OdeMethod::Heun, s))
        .collect();

    let mut ratios = Vec::new();
    for i in 0..errors.len() - 1 {
        if errors[i + 1] > 1e-12 {
            ratios.push((errors[i] / errors[i + 1]).ln() / 2.0f64.ln());
        }
    }
    let order = ratios.iter().sum::<f64>() / ratios.len() as f64;

    assert!(
        (1.85..2.15).contains(&order),
        "Heun should be ~O(h^2) on cosine ODE, estimated order = {order:.3}"
    );
}
