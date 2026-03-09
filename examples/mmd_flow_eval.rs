//! Evaluate flow matching quality using MMD (Maximum Mean Discrepancy).
//!
//! MMD from rkhs provides a kernel-based two-sample test: does the generated
//! distribution match the target? This complements Wasserstein distance
//! (already used in flowmatch) -- MMD is faster and works well in high
//! dimensions where OT becomes expensive.
//!
//! Demonstrates: train a semidiscrete flow, sample from it, then use
//! MMD + permutation test to verify distributional match.
//!
//! Run: cargo run --example mmd_flow_eval

use flowmatch::sd_fm::{SdFmTrainConfig, TimestepSchedule};
use ndarray::{Array1, Array2};
use rkhs::{mmd_unbiased, rbf};
use wass::semidiscrete::SemidiscreteSgdConfig;

fn main() {
    println!("=== MMD Evaluation of Flow Matching ===\n");

    // Target: 4 well-separated points in 2D
    let target =
        Array2::from_shape_vec((4, 2), vec![3.0, 3.0, 3.0, -3.0, -3.0, 3.0, -3.0, -3.0]).unwrap();
    let n = target.nrows();
    let b = Array1::from_elem(n, 1.0 / n as f32);

    // Train semidiscrete flow matching
    let pot_cfg = SemidiscreteSgdConfig::default();
    let fm_cfg = SdFmTrainConfig {
        lr: 0.05,
        steps: 2000,
        batch_size: 256,
        sample_steps: 50,
        seed: 42,
        t_schedule: TimestepSchedule::Uniform,
    };
    let trained = flowmatch::sd_fm::train_sd_fm_semidiscrete_linear(
        &target.view(),
        &b.view(),
        &pot_cfg,
        &fm_cfg,
    )
    .expect("training failed");

    // Generate samples via the trained model's ODE integrator
    let n_gen = 200;
    let (generated_arr, _js) = trained.sample(n_gen, 99, 50).expect("sampling failed");

    // Convert to Vec<Vec<f64>> for rkhs MMD
    let generated: Vec<Vec<f64>> = (0..generated_arr.nrows())
        .map(|i| generated_arr.row(i).iter().map(|&v| v as f64).collect())
        .collect();

    let target_vecs: Vec<Vec<f64>> = (0..target.nrows())
        .map(|i| target.row(i).iter().map(|&v| v as f64).collect())
        .collect();

    // Also generate "noise" baseline (random N(0,1) points)
    let noise: Vec<Vec<f64>> = (0..n_gen)
        .map(|i| {
            // deterministic pseudo-random for reproducibility
            let s = i as f64 * 0.1;
            vec![s.sin() * 3.0, s.cos() * 3.0]
        })
        .collect();

    // MMD at different bandwidths
    println!("MMD^2 (unbiased estimate):\n");
    println!(
        "  {:>10} | {:>18} | {:>18}",
        "bandwidth", "noise vs target", "generated vs target"
    );
    println!("  {:-<10}-+-{:-<18}-+-{:-<18}", "", "", "");

    for &bw in &[0.5, 1.0, 2.0, 5.0] {
        let kernel = |a: &[f64], b: &[f64]| rbf(a, b, bw);
        let mmd_noise = mmd_unbiased(&noise, &target_vecs, kernel);
        let mmd_gen = mmd_unbiased(&generated, &target_vecs, kernel);
        println!("  {:10.1} | {:18.6} | {:18.6}", bw, mmd_noise, mmd_gen);
    }

    // Permutation test: is the generated distribution indistinguishable from target?
    let (mmd_val, p_value) =
        rkhs::mmd_permutation_test(&generated, &target_vecs, |a, b| rbf(a, b, 2.0), 500);
    println!("\nPermutation test (bw=2.0, 500 permutations):");
    println!("  MMD^2 = {mmd_val:.6}, p-value = {p_value:.4}");
    if p_value > 0.05 {
        println!("  -> Cannot reject H0: generated ~ target (flow learned the distribution)");
    } else {
        println!("  -> Reject H0: distributions differ (flow may need more training)");
    }
}
