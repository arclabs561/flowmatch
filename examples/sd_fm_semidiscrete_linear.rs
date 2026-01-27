//! SD-FM (semidiscrete flow matching), minimal runnable demo.
//!
//! Prints:
//! - mean squared distance to the assigned discrete target after sampling
//! - a few assignments / coordinates for sanity

use flowmatch::sd_fm::{train_sd_fm_semidiscrete_linear, SdFmTrainConfig, TimestepSchedule};
use ndarray::{Array1, Array2};
use wass::semidiscrete::SemidiscreteSgdConfig;

fn mean_sq_to_assigned_y(xs: &Array2<f32>, js: &[usize], y: &Array2<f32>) -> f32 {
    let n = xs.nrows();
    let d = xs.ncols();
    let mut s: f64 = 0.0;
    for i in 0..n {
        let j = js[i];
        for k in 0..d {
            let r = (xs[[i, k]] - y[[j, k]]) as f64;
            s += r * r;
        }
    }
    (s / (n as f64 * d as f64)) as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 16usize;
    let d = 8usize;

    let mut y = Array2::<f32>::zeros((n, d));
    for j in 0..n {
        for k in 0..d {
            y[[j, k]] = (((j * 37 + k * 11) % 97) as f32 / 97.0) * 2.0 - 1.0;
        }
    }

    let b = Array1::<f32>::from_vec(vec![1.0; n]);

    let pot_cfg = SemidiscreteSgdConfig {
        epsilon: 0.0,
        lr: 0.8,
        steps: 2_000,
        batch_size: 1_024,
        seed: 7,
    };
    let fm_cfg = SdFmTrainConfig {
        lr: 8e-3,
        steps: 800,
        batch_size: 256,
        sample_steps: 40,
        seed: 9,
        t_schedule: TimestepSchedule::Uniform,
    };

    let trained = train_sd_fm_semidiscrete_linear(&y.view(), &b.view(), &pot_cfg, &fm_cfg)?;

    let n_samp = 64usize;
    let (xs, js) = trained.sample(n_samp, 123, fm_cfg.sample_steps)?;
    let mse = mean_sq_to_assigned_y(&xs, &js, &trained.y);

    println!("n={n} d={d}");
    println!(
        "pot_cfg: steps={} batch={} seed={}",
        pot_cfg.steps, pot_cfg.batch_size, pot_cfg.seed
    );
    println!(
        "fm_cfg:  steps={} batch={} lr={} seed={} euler_steps={}",
        fm_cfg.steps, fm_cfg.batch_size, fm_cfg.lr, fm_cfg.seed, fm_cfg.sample_steps
    );
    println!("sample_mse_to_assigned_y = {mse:.4}");
    println!();

    for i in 0..8.min(n_samp) {
        let j = js[i];
        println!(
            "i={i:2}  j*={j:2}  x1[0..3]=[{:.3}, {:.3}, {:.3}]  yj[0..3]=[{:.3}, {:.3}, {:.3}]",
            xs[[i, 0]],
            xs[[i, 1]],
            xs[[i, 2]],
            trained.y[[j, 0]],
            trained.y[[j, 1]],
            trained.y[[j, 2]],
        );
    }

    Ok(())
}
