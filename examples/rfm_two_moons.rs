//! Two-moons flow matching demo.
//!
//! The standard 2D benchmark: transport samples from N(0,I) to a two-moons
//! distribution using a linear conditional flow field with minibatch OT pairing.
//!
//! Sampling conditions on assigned training targets. A fresh target draw measures
//! how well the output covers both moons at this sample size.
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_two_moons
//! ```

use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
    TimestepSchedule,
};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::error::Error;
use std::ffi::OsStr;
use std::path::Path;

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

type Point = [f32; 2];

fn points(array: &Array2<f32>) -> Vec<Point> {
    array
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1]])
        .collect()
}

fn mean_nearest(from: &[Point], to: &[Point]) -> f64 {
    let sum: f64 = from
        .iter()
        .map(|&[x, y]| {
            to.iter()
                .map(|&[tx, ty]| {
                    let dx = f64::from(x) - f64::from(tx);
                    let dy = f64::from(y) - f64::from(ty);
                    dx * dx + dy * dy
                })
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .sum();
    sum / from.len() as f64
}

fn report(label: &str, samples: &[Point], held_out: &[Point]) {
    println!(
        "{label}: samples -> held-out = {:.4}, held-out -> samples = {:.4}",
        mean_nearest(samples, held_out),
        mean_nearest(held_out, samples)
    );
}

fn parse_points(bytes: &[u8]) -> Result<Vec<Point>, Box<dyn Error>> {
    let points: Vec<Point> = serde_json::from_slice(bytes)?;
    if points.is_empty() || points.iter().flatten().any(|v| !v.is_finite()) {
        return Err("expected a nonempty array of finite [x, y] points".into());
    }
    Ok(points)
}

fn read_points(path: &Path) -> Result<Vec<Point>, Box<dyn Error>> {
    let bytes = std::fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
    parse_points(&bytes).map_err(|err| format!("{}: {err}", path.display()).into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [] => (),
        [flag, samples, held_out] if flag == OsStr::new("--compare") => {
            report(
                "input",
                &read_points(Path::new(samples))?,
                &read_points(Path::new(held_out))?,
            );
            return Ok(());
        }
        _ => return Err("usage: rfm_two_moons [--compare SAMPLES.json HELD_OUT.json]".into()),
    }
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

    let held_out = points(&sample_two_moons(n_target, noise, 43));
    let fresh = points(&sample_two_moons(n_samples, noise, 44));
    let collapsed = vec![[target[[0, 0]], target[[0, 1]]]; n_samples];
    let mut rng = ChaCha8Rng::seed_from_u64(999);
    let gaussian: Vec<Point> = (0..n_samples)
        .map(|_| {
            [
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ]
        })
        .collect();

    println!(
        "Two moons (training={n_target}, held-out={n_target}, samples={n_samples}, noise={noise})"
    );
    println!("Mean nearest-neighbor distances (lower is better in each direction)");
    report("Gaussian", &gaussian, &held_out);
    report("one training point repeated", &collapsed, &held_out);
    report("fresh moons", &fresh, &held_out);
    report("trained (endpoint-conditioned)", &points(&xs), &held_out);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_samples_miss_a_mode_in_the_reverse_direction() {
        let targets = [[0., 0.], [3., 4.]];
        let repeated = [[0., 0.]; 3];
        assert_eq!(mean_nearest(&repeated, &targets), 0.);
        assert_eq!(mean_nearest(&targets, &repeated), 2.5);
        assert_eq!(mean_nearest(&targets, &targets), 0.);
    }

    #[test]
    fn point_files_require_nonempty_finite_pairs() {
        assert_eq!(parse_points(b"[[0,1.5]]").unwrap(), [[0.0, 1.5]]);
        for text in ["[]", "[[0]]", "[[0,1,2]]", "[[0,1e100]]"] {
            assert!(parse_points(text.as_bytes()).is_err(), "{text}");
        }
    }
}
