//! Timing breakdown for the torsions (φ/ψ) RFM loop.
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example profile_breakdown_torsions
//! ```

use flowmatch::linear::LinearCondField;
use flowmatch::rfm::{minibatch_ot_greedy_pairing, minibatch_rowwise_nearest_pairing};
use flowmatch::sd_fm::RfmMinibatchOtConfig;
use flowmatch::{Error, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::time::{Duration, Instant};

const PHI_PSI_CSV: &str = include_str!("../examples_data/pdb_1bpi_phi_psi.csv.txt");

fn embed_phi_psi(phi: f32, psi: f32) -> [f32; 4] {
    [phi.cos(), phi.sin(), psi.cos(), psi.sin()]
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
    let t0 = Instant::now();
    let mut phi_psi: Vec<(f32, f32)> = Vec::new();
    for (line_idx, line) in PHI_PSI_CSV.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        let phi: f32 = parts[4].parse().unwrap_or(0.0);
        let psi: f32 = parts[5].parse().unwrap_or(0.0);
        if phi.is_finite() && psi.is_finite() {
            phi_psi.push((phi, psi));
        }
    }
    let load_time = t0.elapsed();

    if phi_psi.len() < 20 {
        return Err(Error::Domain("not enough parsed phi/psi pairs"));
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
    let b_norm: Vec<f32> = vec![1.0 / (n as f32); n];

    // Config
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

        // 2) Sample y indices uniformly.
        let t = Instant::now();
        for i in 0..batch_size {
            // uniform categorical
            let j = (rng.random::<f32>() * (n as f32))
                .floor()
                .min((n - 1) as f32) as usize;
            let yj = y.row(j);
            for k in 0..d {
                ys[[i, k]] = yj[k];
            }
            let _ = b_norm[j]; // keep the contract explicit: y is uniform-weighted here
        }
        timings.sample_y += t.elapsed();

        // 3) Pairing.
        let t = Instant::now();
        let perm = minibatch_ot_greedy_pairing(
            &x0s.view(),
            &ys.view(),
            rfm_cfg.reg,
            rfm_cfg.max_iter,
            rfm_cfg.tol,
        )?;
        timings.sinkhorn_pair += t.elapsed();

        // 3b) Fast pairing (rowwise nearest). Purely for profiling comparison.
        let t = Instant::now();
        let _perm_fast = minibatch_rowwise_nearest_pairing(&x0s.view(), &ys.view())?;
        timings.fast_pair += t.elapsed();

        // 4) SGD.
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
    let total = timings.sample_x0 + timings.sample_y + timings.sinkhorn_pair + timings.sgd;

    println!("Torsions profile breakdown (PDB 1BPI-derived)");
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
