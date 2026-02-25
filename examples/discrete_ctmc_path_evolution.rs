//! Discrete Flow Matching: probability path evolution under different schedules.
//!
//! Reproduces the core idea from Gat et al. (2024, NeurIPS): a CTMC drives
//! a categorical distribution from a source state x_0 toward a target state x_1,
//! with the transition rate controlled by an interpolation schedule kappa(t).
//!
//! This example shows:
//! 1. How the conditional probability path p_t(x | x_0, x_1) evolves for each schedule.
//! 2. How the conditional rate matrix R_t concentrates flow on the x_0 -> x_1 transition.
//! 3. How forward Euler integration of dp/dt = p R_t recovers the analytical path.
//!
//! Run: cargo run -p flowmatch --example discrete_ctmc_path_evolution

use flowmatch::discrete_ctmc::{
    conditional_probability_path, conditional_rate_matrix, CtmcGenerator, DiscreteSchedule,
};

fn main() {
    let k = 4; // number of states
    let x0 = 0; // source state
    let x1 = 2; // target state
    let eps = 1e-5;
    let schedules = [
        ("Linear", DiscreteSchedule::Linear),
        ("CosineSq", DiscreteSchedule::CosineSq),
        ("CosineHalf", DiscreteSchedule::CosineHalf),
    ];

    println!("=== Discrete Flow Matching: Probability Path Evolution ===");
    println!("States: k={k}, source: x0={x0}, target: x1={x1}\n");

    // --- Part 1: Analytical conditional probability path ---
    println!("--- Analytical p_t(x | x0, x1) ---\n");
    println!("{:>10}  {:>8} {:>8} {:>8} {:>8}", "t", "p[0]", "p[1]", "p[2]", "p[3]");
    println!("{:-<10}  {:-<8} {:-<8} {:-<8} {:-<8}", "", "", "", "", "");

    for &(name, sched) in &schedules {
        println!("\n  Schedule: {name}");
        for &t in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let p = conditional_probability_path(sched, t, x0, x1, k).unwrap();
            println!(
                "{:>10.2}  {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
                t, p[0], p[1], p[2], p[3]
            );
        }
    }

    // --- Part 2: Rate matrix structure at t=0.3 ---
    println!("\n\n--- Conditional rate matrix R_t at t=0.3 ---\n");
    for &(name, sched) in &schedules {
        let r = conditional_rate_matrix(sched, 0.3, x0, x1, k, eps).unwrap();
        let kd = sched.kappa_dot(0.3);
        let kv = sched.kappa(0.3);
        println!("  {name}: kappa(0.3)={kv:.4}, kappa'(0.3)={kd:.4}");
        println!("  Rate x0->x1 = {:.4}, diagonal = {:.4}\n", r[[x0, x1]], r[[x0, x0]]);
    }

    // --- Part 3: Forward Euler integration vs analytical path ---
    println!("--- Euler integration vs analytical (CosineSq schedule) ---\n");
    let sched = DiscreteSchedule::CosineSq;
    let n_steps = 1000;
    let dt = 1.0 / n_steps as f32;

    // Start at one-hot on x0
    let mut p_euler = ndarray::Array1::zeros(k);
    p_euler[x0] = 1.0;

    println!("{:>6}  {:>24}  {:>24}  {:>8}", "t", "Euler p[x0], p[x1]", "Exact p[x0], p[x1]", "L1 err");
    println!("{:-<6}  {:-<24}  {:-<24}  {:-<8}", "", "", "", "");

    let checkpoints = [0, 100, 250, 500, 750, 900, 999];
    for step in 0..n_steps {
        let t = step as f32 * dt;
        // Build rate matrix at current time
        let r = conditional_rate_matrix(sched, t, x0, x1, k, eps).unwrap();
        let gen = CtmcGenerator { q: r };
        p_euler = gen.step_euler(&p_euler.view(), dt).unwrap();

        if checkpoints.contains(&step) {
            let t_next = (step + 1) as f32 * dt;
            let p_exact = conditional_probability_path(sched, t_next, x0, x1, k).unwrap();
            let l1: f32 = p_euler
                .iter()
                .zip(p_exact.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            println!(
                "{:>6.3}  {:>11.4}, {:>11.4}  {:>11.4}, {:>11.4}  {:>8.2e}",
                t_next, p_euler[x0], p_euler[x1], p_exact[x0], p_exact[x1], l1
            );
        }
    }

    // --- Part 4: Schedule comparison (kappa and kappa') ---
    println!("\n\n--- Schedule profiles: kappa(t) and kappa'(t) ---\n");
    println!("{:>6}  {:>18}  {:>18}  {:>18}", "t", "Linear", "CosineSq", "CosineHalf");
    println!("{:-<6}  {:-<18}  {:-<18}  {:-<18}", "", "", "", "");

    for i in 0..=10 {
        let t = i as f32 / 10.0;
        let kl = DiscreteSchedule::Linear.kappa(t);
        let kc = DiscreteSchedule::CosineSq.kappa(t);
        let kh = DiscreteSchedule::CosineHalf.kappa(t);
        let kdl = DiscreteSchedule::Linear.kappa_dot(t);
        let kdc = DiscreteSchedule::CosineSq.kappa_dot(t);
        let kdh = DiscreteSchedule::CosineHalf.kappa_dot(t);
        println!(
            "{:>6.1}  {:>8.4} ({:>6.3})  {:>8.4} ({:>6.3})  {:>8.4} ({:>6.3})",
            t, kl, kdl, kc, kdc, kh, kdh
        );
    }

    println!("\nKey insight from Gat et al. (2024):");
    println!("  The cosine-squared schedule has kappa'(0) = kappa'(1) = 0, avoiding");
    println!("  the 1/(1-t) singularity in the rate matrix near t=1. The linear schedule");
    println!("  has constant kappa'=1, causing the rate to blow up as t -> 1.");
}
