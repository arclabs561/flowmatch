//! Timing breakdown for the USGS “sphere-ish” RFM loop.
//!
//! This is a pragmatic “profile” when we don't have a working CPU-profiler toolchain:
//! it tells you where time is going (sampling vs Sinkhorn vs SGD).
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example profile_breakdown_usgs
//! ```

use flowmatch::linear::LinearCondField;
use flowmatch::rfm::minibatch_ot_greedy_pairing;
use flowmatch::rfm::minibatch_rowwise_nearest_pairing;
use flowmatch::sd_fm::RfmMinibatchOtConfig;
use flowmatch::{Error, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::time::{Duration, Instant};

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

fn sample_categorical_from_probs(probs: &[f32], rng: &mut impl rand::Rng) -> usize {
    let u: f32 = rng.random();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u <= acc {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

#[derive(Default, Debug, Clone)]
struct Timings {
    sample_x0: Duration,
    sample_y: Duration,
    sinkhorn_pair: Duration,
    fast_pair: Duration,
    sgd: Duration,
}

fn main() -> Result<()> {
    // Load dataset.
    let t0 = Instant::now();
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
    let load_time = t0.elapsed();

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
    let bs = b.sum();
    let b_norm: Vec<f32> = b.iter().map(|&x| x / bs).collect();

    // Config (tweak these to profile bigger/smaller runs).
    let steps = 600usize;
    let batch_size = 64usize;
    let lr = 2e-2f32;
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 0.2,
        max_iter: 2_000,
        tol: 2e-3,
        pairing: flowmatch::sd_fm::RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 1,
    };
    let seed = 123u64;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut field = LinearCondField::new_zeros(d);

    let mut x0s = Array2::<f32>::zeros((batch_size, d));
    let mut ys = Array2::<f32>::zeros((batch_size, d));

    let mut timings = Timings::default();
    let t_train0 = Instant::now();

    for _step in 0..steps {
        // 1) Sample x0.
        let t = Instant::now();
        for i in 0..batch_size {
            for k in 0..d {
                x0s[[i, k]] = StandardNormal.sample(&mut rng);
            }
        }
        timings.sample_x0 += t.elapsed();

        // 2) Sample y indices by b.
        let t = Instant::now();
        for i in 0..batch_size {
            let j = sample_categorical_from_probs(&b_norm, &mut rng);
            let yj = y.row(j);
            for k in 0..d {
                ys[[i, k]] = yj[k];
            }
        }
        timings.sample_y += t.elapsed();

        // 3) Minibatch OT pairing (Sinkhorn).
        let t = Instant::now();
        let perm = minibatch_ot_greedy_pairing(
            &x0s.view(),
            &ys.view(),
            rfm_cfg.reg,
            rfm_cfg.max_iter,
            rfm_cfg.tol,
        )?;
        timings.sinkhorn_pair += t.elapsed();

        // 3b) Fast pairing (rowwise nearest). This is purely for profiling comparison.
        let t = Instant::now();
        let _perm_fast = minibatch_rowwise_nearest_pairing(&x0s.view(), &ys.view())?;
        timings.fast_pair += t.elapsed();

        // 4) FM regression updates.
        let t = Instant::now();
        for (i, &p) in perm.iter().enumerate().take(batch_size) {
            let x0 = x0s.row(i);
            let y1 = ys.row(p);

            let tt: f32 = rng.random();
            let mut xt = Array1::<f32>::zeros(d);
            for k in 0..d {
                xt[k] = (1.0 - tt) * x0[k] + tt * y1[k];
            }
            let mut u = Array1::<f32>::zeros(d);
            for k in 0..d {
                u[k] = y1[k] - x0[k];
            }

            field.sgd_step(&xt.view(), tt, &y1, &u.view(), lr);
        }
        timings.sgd += t.elapsed();
    }

    let train_time = t_train0.elapsed();
    // Note: `fast_pair` is *additional* work; don't include it in accounted total.
    let total = timings.sample_x0 + timings.sample_y + timings.sinkhorn_pair + timings.sgd;

    println!("USGS profile breakdown");
    println!("- n_support={n} d={d} steps={steps} batch_size={batch_size}");
    println!("- load_time: {:?}", load_time);
    println!("- train_time: {:?}", train_time);
    println!(
        "- accounted_total: {:?} (should be close to train_time)",
        total
    );

    let denom = total.as_secs_f64().max(1e-12);
    for (name, dur) in [
        ("sample_x0", timings.sample_x0),
        ("sample_y", timings.sample_y),
        ("sinkhorn_pair", timings.sinkhorn_pair),
        ("fast_pair", timings.fast_pair),
        ("sgd", timings.sgd),
    ] {
        println!(
            "  - {:>12}: {:>10.3} ms ({:>5.1}%)",
            name,
            1e3 * dur.as_secs_f64(),
            100.0 * dur.as_secs_f64() / denom
        );
    }

    Ok(())
}
