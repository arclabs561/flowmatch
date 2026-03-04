//! Two-moons flow matching demo.
//!
//! The standard 2D benchmark: transport samples from N(0,I) to a two-moons
//! distribution using a linear conditional flow field with minibatch OT pairing.
//!
//! This is the minimal "does it work on a multimodal target?" sanity check.
//! The target is synthetic (no external data), so the example is self-contained.
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_two_moons
//! ```

use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
    TimestepSchedule,
};
use flowmatch::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

/// Generate `n` samples from a two-moons distribution.
///
/// Upper moon: semicircle centered at (0, 0), radius 1, angles in [0, pi], with Gaussian noise.
/// Lower moon: semicircle centered at (1, 0.5), radius 1, angles in [pi, 2*pi], with Gaussian noise.
fn sample_two_moons(n: usize, noise: f32, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0f32, noise).unwrap();
    let mut out = Array2::<f32>::zeros((n, 2));
    let pi = core::f32::consts::PI;
    for i in 0..n {
        let u: f32 = rng.random();
        let angle = u * pi;
        if i % 2 == 0 {
            // Upper moon
            out[[i, 0]] = angle.cos() + noise_dist.sample(&mut rng);
            out[[i, 1]] = angle.sin() + noise_dist.sample(&mut rng);
        } else {
            // Lower moon (shifted)
            out[[i, 0]] = 1.0 - angle.cos() + noise_dist.sample(&mut rng);
            out[[i, 1]] = 0.5 - angle.sin() + noise_dist.sample(&mut rng);
        }
    }
    out
}

/// Squared L2 distance between two 2D points.
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn main() -> Result<()> {
    let n_target = 500;
    let noise = 0.05f32;
    let target = sample_two_moons(n_target, noise, 42);

    // Uniform weights (all target points equally important).
    let b = Array1::<f32>::from_elem(n_target, 1.0);

    let fm_cfg = SdFmTrainConfig {
        lr: 5e-3,
        steps: 3_000,
        batch_size: 128,
        sample_steps: 30,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 0.5,
        max_iter: 2_000,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 1,
    };

    let model = train_rfm_minibatch_ot_linear(&target.view(), &b.view(), &rfm_cfg, &fm_cfg)?;

    // Sample from the trained flow.
    let n_samples = 500;
    let (xs, _js) = model.sample(n_samples, 777, fm_cfg.sample_steps)?;

    // Evaluate: for each generated sample, find the nearest target point.
    // Report mean nearest-neighbor distance (a simple proxy for coverage).
    let mut nn_dists: Vec<f32> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let si = [xs[[i, 0]], xs[[i, 1]]];
        let mut best = f32::INFINITY;
        for j in 0..n_target {
            let tj = [target[[j, 0]], target[[j, 1]]];
            let d = sq_dist(&si, &tj);
            if d < best {
                best = d;
            }
        }
        nn_dists.push(best.sqrt());
    }
    nn_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_nn: f32 = nn_dists.iter().sum::<f32>() / (n_samples as f32);
    let median_nn: f32 = nn_dists[n_samples / 2];

    // Baseline: raw Gaussian samples (no training).
    let mut rng = ChaCha8Rng::seed_from_u64(999);
    let mut baseline_nn: Vec<f32> = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let s0: f32 = StandardNormal.sample(&mut rng);
        let s1: f32 = StandardNormal.sample(&mut rng);
        let si = [s0, s1];
        let mut best = f32::INFINITY;
        for j in 0..n_target {
            let tj = [target[[j, 0]], target[[j, 1]]];
            let d = sq_dist(&si, &tj);
            if d < best {
                best = d;
            }
        }
        baseline_nn.push(best.sqrt());
    }
    baseline_nn.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let base_mean: f32 = baseline_nn.iter().sum::<f32>() / (n_samples as f32);
    let base_median: f32 = baseline_nn[n_samples / 2];

    println!("Two-moons flow matching (n_target={n_target}, n_samples={n_samples}, noise={noise})");
    println!("Nearest-neighbor distance to target (lower is better):");
    println!("- baseline (N(0,I)):  mean={base_mean:.4}  median={base_median:.4}");
    println!("- trained  (RFM+OT): mean={mean_nn:.4}  median={median_nn:.4}");
    println!(
        "- ratio mean trained/baseline: {:.3}",
        mean_nn / base_mean.max(1e-12)
    );

    // Print a few sample trajectories (initial x0, final x1) for inspection.
    let (x0s, x1s, js) = model.sample_with_x0(10, 888, fm_cfg.sample_steps)?;
    println!("\nSample trajectories (x0 -> x1, conditioned on target j):");
    for i in 0..10.min(x1s.nrows()) {
        println!(
            "  [{:>6.3}, {:>6.3}] -> [{:>6.3}, {:>6.3}]  (j={})",
            x0s[[i, 0]],
            x0s[[i, 1]],
            x1s[[i, 0]],
            x1s[[i, 1]],
            js[i],
        );
    }

    Ok(())
}
