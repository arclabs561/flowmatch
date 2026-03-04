//! Timing breakdown for the USGS "sphere-ish" RFM loop.
//!
//! This is a pragmatic "profile" when we don't have a working CPU-profiler toolchain:
//! it tells you where time is going (sampling vs Sinkhorn vs SGD).
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example profile_breakdown_usgs
//! ```

mod common;

use common::usgs::{build_support_and_weights, parse_usgs_csv};
use flowmatch::linear::LinearCondField;
use flowmatch::rfm::minibatch_ot_greedy_pairing;
use flowmatch::rfm::minibatch_rowwise_nearest_pairing;
use flowmatch::sd_fm::RfmMinibatchOtConfig;
use flowmatch::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::time::Instant;

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

use common::Timings;

fn main() -> Result<()> {
    let t0 = Instant::now();
    let data = parse_usgs_csv(10)?;
    let load_time = t0.elapsed();

    let n = data.pts.len();
    let d = 3usize;
    let (y, b) = build_support_and_weights(&data);

    let bs = b.sum();
    let b_norm: Vec<f32> = b.iter().map(|&x| x / bs).collect();

    // Config (tweak these to profile bigger/smaller runs).
    let steps = 600usize;
    let batch_size = 64usize;
    let lr = 2e-2f32;
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 1.0,
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
    println!("USGS profile breakdown");
    println!("- n_support={n} d={d} steps={steps} batch_size={batch_size}");
    println!("- load_time: {:?}", load_time);
    println!("- train_time: {:?}", train_time);
    println!(
        "- accounted_total: {:?} (should be close to train_time)",
        timings.accounted_total()
    );
    timings.print();

    Ok(())
}
