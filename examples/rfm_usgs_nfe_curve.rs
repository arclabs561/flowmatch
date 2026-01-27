//! USGS RFM: quality vs sampling steps (NFE curve).
//!
//! This is closer to what rectified-flow papers actually report:
//! sample quality as a function of integration steps / NFE.
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_usgs_nfe_curve
//! ```
//!
//! Optional:
//! ```bash
//! # U-shaped timestep sampling (more weight near t=0 and t=1)
//! FLOWMATCH_T_SCHEDULE=ushaped cargo run -p flowmatch --example rfm_usgs_nfe_curve
//! ```

use flowmatch::metrics::ot_cost_samples_to_weighted_support;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
    TimestepSchedule,
};
use flowmatch::{Error, Result};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

const USGS_CSV: &str = include_str!("../examples_data/usgs_eq_m6_2024_limit50.csv.txt");

fn deg_to_rad(x: f32) -> f32 {
    x * core::f32::consts::PI / 180.0
}

fn latlon_to_unit_xyz(lat_deg: f32, lon_deg: f32) -> [f32; 3] {
    let lat = deg_to_rad(lat_deg);
    let lon = deg_to_rad(lon_deg);
    let clat = lat.cos();
    [clat * lon.cos(), clat * lon.sin(), lat.sin()]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 0.0 && n.is_finite() {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len().max(1) as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn main() -> Result<()> {
    // Load USGS data.
    let mut pts: Vec<[f32; 3]> = Vec::new();
    let mut mags: Vec<f32> = Vec::new();
    for (line_idx, line) in USGS_CSV.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        let lat: f32 = parts[1].parse().unwrap_or(0.0);
        let lon: f32 = parts[2].parse().unwrap_or(0.0);
        let mag: f32 = parts[4].parse().unwrap_or(0.0);
        if !lat.is_finite() || !lon.is_finite() || !mag.is_finite() {
            continue;
        }
        pts.push(latlon_to_unit_xyz(lat, lon));
        mags.push(mag);
    }
    if pts.len() < 10 {
        return Err(Error::Domain("not enough parsed USGS points"));
    }

    let n = pts.len();
    let d = 3usize;
    let mut y = Array2::<f32>::zeros((n, d));
    for (i, p) in pts.iter().enumerate() {
        y[[i, 0]] = p[0];
        y[[i, 1]] = p[1];
        y[[i, 2]] = p[2];
    }
    let mut b = Array1::<f32>::zeros(n);
    for i in 0..n {
        b[i] = (mags[i] - 5.0).max(0.0).exp();
    }

    let t_schedule = match std::env::var("FLOWMATCH_T_SCHEDULE").as_deref() {
        Ok("ushaped") => TimestepSchedule::UShaped,
        _ => TimestepSchedule::Uniform,
    };

    // Train model.
    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: 2_500,
        batch_size: 64,
        sample_steps: 0,
        seed: 123,
        t_schedule,
    };
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 0.2,
        max_iter: 2_000,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 2,
    };
    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;

    let m = 1_500usize;
    let seeds = [900u64, 901, 902, 903, 904];
    let mut baseline_vals: Vec<f64> = Vec::new();
    for &seed in &seeds {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut xs0 = Array2::<f32>::zeros((m, d));
        for i in 0..m {
            let v = normalize3([
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ]);
            xs0[[i, 0]] = v[0];
            xs0[[i, 1]] = v[1];
            xs0[[i, 2]] = v[2];
        }
        let ot = ot_cost_samples_to_weighted_support(
            &xs0.view(),
            &y.view(),
            &(&b / b.sum()).view(),
            0.10,
            8_000,
            2e-3,
        )?;
        baseline_vals.push(ot as f64);
    }
    let (baseline_mean, baseline_std) = mean_std(&baseline_vals);

    println!("USGS NFE curve (n_support={n}, d={d})");
    println!("train t_schedule={t_schedule:?}");
    println!("baseline OT cost: mean={baseline_mean:.4}  std={baseline_std:.4}");
    println!();

    for steps in [1usize, 2, 4, 8, 16, 32] {
        let mut vals: Vec<f64> = Vec::new();
        for &seed in &seeds {
            let (xs_raw, _js) = model.sample(m, seed, steps)?;
            let mut xs = Array2::<f32>::zeros((m, d));
            for i in 0..m {
                let v = normalize3([xs_raw[[i, 0]], xs_raw[[i, 1]], xs_raw[[i, 2]]]);
                xs[[i, 0]] = v[0];
                xs[[i, 1]] = v[1];
                xs[[i, 2]] = v[2];
            }
            let ot = ot_cost_samples_to_weighted_support(
                &xs.view(),
                &y.view(),
                &(&b / b.sum()).view(),
                0.10,
                8_000,
                2e-3,
            )?;
            vals.push(ot as f64);
        }
        let (mean, std) = mean_std(&vals);
        println!(
            "steps={:>2}  ot_cost: mean={:.4} std={:.4}  ratio_vs_baseline_mean={:.3}",
            steps,
            mean,
            std,
            mean / baseline_mean.max(1e-12),
        );
    }

    Ok(())
}
