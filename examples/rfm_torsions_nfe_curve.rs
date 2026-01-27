//! Protein torsions RFM: NFE/steps curve (seed-averaged).
//!
//! Mirrors the USGS NFE curve, but on the torsion-angle (torus) dataset.
//! Metric: Ramachandran histogram JS divergence (lower is better).
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_torsions_nfe_curve
//! ```
//!
//! Optional:
//! ```bash
//! FLOWMATCH_T_SCHEDULE=ushaped cargo run -p flowmatch --example rfm_torsions_nfe_curve
//! ```

use flowmatch::metrics::jensen_shannon_divergence_histogram;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
    TimestepSchedule,
};
use flowmatch::{Error, Result};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

const PHI_PSI_CSV: &str = include_str!("../examples_data/pdb_1bpi_phi_psi.csv.txt");

fn wrap_pi(x: f32) -> f32 {
    let mut y = x % (2.0 * core::f32::consts::PI);
    if y <= -core::f32::consts::PI {
        y += 2.0 * core::f32::consts::PI;
    }
    if y > core::f32::consts::PI {
        y -= 2.0 * core::f32::consts::PI;
    }
    y
}

fn embed_phi_psi(phi: f32, psi: f32) -> [f32; 4] {
    [phi.cos(), phi.sin(), psi.cos(), psi.sin()]
}

fn decode_phi_psi(e: [f32; 4]) -> (f32, f32) {
    let phi = e[1].atan2(e[0]);
    let psi = e[3].atan2(e[2]);
    (wrap_pi(phi), wrap_pi(psi))
}

fn rama_hist(phi_psi: &[(f32, f32)], bins: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; bins * bins];
    let two_pi = 2.0 * core::f32::consts::PI;
    for &(phi, psi) in phi_psi {
        let u = (wrap_pi(phi) + core::f32::consts::PI) / two_pi;
        let v = (wrap_pi(psi) + core::f32::consts::PI) / two_pi;
        let mut i = (u * bins as f32).floor() as isize;
        let mut j = (v * bins as f32).floor() as isize;
        if i < 0 {
            i = 0;
        }
        if j < 0 {
            j = 0;
        }
        if i >= bins as isize {
            i = bins as isize - 1;
        }
        if j >= bins as isize {
            j = bins as isize - 1;
        }
        h[(i as usize) * bins + (j as usize)] += 1.0;
    }
    let sum: f32 = h.iter().sum();
    if sum > 0.0 {
        for x in &mut h {
            *x /= sum;
        }
    }
    h
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len().max(1) as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn main() -> Result<()> {
    // Load torsion data.
    let mut phi_psi: Vec<(f32, f32)> = Vec::new();
    for (line_idx, line) in PHI_PSI_CSV.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        let phi: f32 = parts[0].parse().unwrap_or(0.0);
        let psi: f32 = parts[1].parse().unwrap_or(0.0);
        if phi.is_finite() && psi.is_finite() {
            phi_psi.push((wrap_pi(phi), wrap_pi(psi)));
        }
    }
    if phi_psi.len() < 20 {
        return Err(Error::Domain("not enough torsion points parsed"));
    }

    let n = phi_psi.len();
    let d = 4usize;
    let mut y = Array2::<f32>::zeros((n, d));
    for (i, (phi, psi)) in phi_psi.iter().copied().enumerate() {
        let e = embed_phi_psi(phi, psi);
        for k in 0..d {
            y[[i, k]] = e[k];
        }
    }
    let b = Array1::<f32>::from_elem(n, 1.0);

    let t_schedule = match std::env::var("FLOWMATCH_T_SCHEDULE").as_deref() {
        Ok("ushaped") => TimestepSchedule::UShaped,
        _ => TimestepSchedule::Uniform,
    };

    // Train model.
    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: 3_000,
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

    let bins = 16usize;
    let target_hist = rama_hist(&phi_psi, bins);
    let seeds = [900u64, 901, 902, 903, 904];
    let m = 1_500usize;

    // Baseline: random angles uniform via Gaussian in embedding then decode.
    let mut baseline_vals: Vec<f64> = Vec::new();
    for &seed in &seeds {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut samples: Vec<(f32, f32)> = Vec::with_capacity(m);
        for _ in 0..m {
            // Random in embedding space then decode; not uniform on torus, but it's a stable baseline.
            let e = [
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ];
            let (phi, psi) = decode_phi_psi([e[0], e[1], e[2], e[3]]);
            samples.push((phi, psi));
        }
        let h = rama_hist(&samples, bins);
        let js = jensen_shannon_divergence_histogram(&target_hist, &h, 1e-6)? as f64;
        baseline_vals.push(js);
    }
    let (base_mean, base_std) = mean_std(&baseline_vals);

    println!("Torsions NFE curve (n_support={n}, d={d})");
    println!("train t_schedule={t_schedule:?}");
    println!("baseline Ramachandran JS: mean={base_mean:.4}  std={base_std:.4}");
    println!();

    for steps in [1usize, 2, 4, 8, 16, 32] {
        let mut vals: Vec<f64> = Vec::new();
        for &seed in &seeds {
            let (xs_raw, _js) = model.sample(m, seed, steps)?;
            let mut samples: Vec<(f32, f32)> = Vec::with_capacity(m);
            for i in 0..m {
                let (phi, psi) = decode_phi_psi([
                    xs_raw[[i, 0]],
                    xs_raw[[i, 1]],
                    xs_raw[[i, 2]],
                    xs_raw[[i, 3]],
                ]);
                samples.push((phi, psi));
            }
            let h = rama_hist(&samples, bins);
            let js = jensen_shannon_divergence_histogram(&target_hist, &h, 1e-6)? as f64;
            vals.push(js);
        }
        let (mean, std) = mean_std(&vals);
        println!(
            "steps={:>2}  rama_js: mean={:.4} std={:.4}  ratio_vs_baseline_mean={:.3}",
            steps,
            mean,
            std,
            mean / base_mean.max(1e-12)
        );
    }

    Ok(())
}
