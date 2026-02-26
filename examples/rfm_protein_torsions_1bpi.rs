//! Rectified flow matching on **real protein torsion data** (a torus-shaped domain).
//!
//! Motivation (from "Flow Matching on General Geometries" and FoldFlow-style eval culture):
//! torsion angles live on a product of circles \(S^1 \times S^1\) (a 2D torus), and *distribution*
//! matching should be measurable (not just "the code runs").
//!
//! We stay within `flowmatch`'s current primitive (linear conditional field + minibatch OT pairing),
//! but we use **real phi/psi** angles extracted from a PDB structure and score matching via a simple,
//! interpretable distributional metric: **JS divergence** between Ramachandran histograms
//! (computed via the ecosystem `logp` crate through `flowmatch::metrics`).
//!
//! Data provenance:
//! - PDB: `1BPI` chain A (BPTI). Source: RCSB PDB (`https://files.rcsb.org/download/1BPI.pdb`)
//! - This repo vendors a tiny derived artifact: phi/psi angles (radians) computed from backbone atoms.
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_protein_torsions_1bpi
//! ```

mod common;

use common::torsions::{
    build_torsion_support, decode_phi_psi, parse_phi_psi_csv_6col, rama_hist,
};
use flowmatch::linear::LinearCondField;
use flowmatch::metrics::jensen_shannon_divergence_histogram;
use flowmatch::rfm::minibatch_ot_greedy_pairing;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use flowmatch::Result;
use ndarray::{Array2, ArrayView2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Compute mean training loss (MSE) for a minibatch, using the same straight-line
/// flow matching objective as the training loop.
fn estimate_training_mse(
    field: &LinearCondField,
    y: &ArrayView2<f32>,
    b_norm: &[f32],
    batch_size: usize,
    seed: u64,
) -> f32 {
    let d = y.ncols();
    let n = y.nrows();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Sample a minibatch of (x0, y_j) pairs.
    let mut x0s = Array2::<f32>::zeros((batch_size, d));
    let mut ys = Array2::<f32>::zeros((batch_size, d));
    for i in 0..batch_size {
        for k in 0..d {
            x0s[[i, k]] = StandardNormal.sample(&mut rng);
        }
        // sample j ~ b_norm
        let u: f32 = rng.random();
        let mut acc = 0.0f32;
        let mut j = n - 1;
        for (idx, &p) in b_norm.iter().enumerate() {
            acc += p;
            if u < acc {
                j = idx;
                break;
            }
        }
        let yj = y.row(j);
        for k in 0..d {
            ys[[i, k]] = yj[k];
        }
    }

    // OT pairing on this minibatch (same as the training loop).
    let perm =
        minibatch_ot_greedy_pairing(&x0s.view(), &ys.view(), 1.0, 2_000, 2e-3).unwrap_or_else(
            |_| (0..batch_size).collect(),
        );

    let mut ts = Vec::with_capacity(batch_size);
    let mut us = Array2::<f32>::zeros((batch_size, d));
    let mut xts = Array2::<f32>::zeros((batch_size, d));
    let mut y_paired = Array2::<f32>::zeros((batch_size, d));

    for (i, &p) in perm.iter().enumerate().take(batch_size) {
        let t: f32 = rng.random();
        ts.push(t);
        for k in 0..d {
            let x0k = x0s[[i, k]];
            let y1k = ys[[p, k]];
            xts[[i, k]] = (1.0 - t) * x0k + t * y1k;
            us[[i, k]] = y1k - x0k;
            y_paired[[i, k]] = y1k;
        }
    }

    let mse = field.mse_batch(&xts.view(), &ts, &y_paired.view(), &us.view());
    mse
}

fn main() -> Result<()> {
    let phi_psi = parse_phi_psi_csv_6col(20)?;
    let n = phi_psi.len();
    let (y, b) = build_torsion_support(&phi_psi);

    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: 3_000,
        batch_size: 64,
        sample_steps: 40,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 1.0,
        max_iter: 2_000,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 1,
    };

    // Empirical histogram.
    let bins = 36usize;
    let h_data = rama_hist(&phi_psi, bins);

    // Baseline: Gaussian -> decode as angles via atan2(sin,cos).
    let baseline_js = {
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let mut samples: Vec<(f32, f32)> = Vec::new();
        for _ in 0..512 {
            let e = [
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ];
            samples.push(decode_phi_psi(e));
        }
        let h0 = rama_hist(&samples, bins);
        jensen_shannon_divergence_histogram(&h_data, &h0, 1e-6)?
    };

    // Train with periodic loss reporting.
    // We train in chunks and report MSE at each checkpoint.
    let b_norm_vec: Vec<f32> = {
        let bs: f32 = b.iter().sum();
        b.iter().map(|&x| x / bs).collect()
    };
    let checkpoints = [500usize, 1000, 1500, 2000, 2500, 3000];
    println!("Training loss (MSE on a held-out minibatch of 256):");

    // Train the full model first, then report.
    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;

    // Report training loss at the trained model (end of training).
    // For intermediate losses, we train partial models at each checkpoint.
    for &ckpt in &checkpoints {
        let partial_cfg = SdFmTrainConfig {
            steps: ckpt,
            ..fm_cfg.clone()
        };
        let partial_model =
            train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &partial_cfg)?;
        let mse = estimate_training_mse(&partial_model.field, &y.view(), &b_norm_vec, 256, 42);
        println!("  step={:>5}  mse={:.6}", ckpt, mse);
    }

    let (xs, _js) = model.sample(512, 777, fm_cfg.sample_steps)?;

    let trained_js = {
        let mut samples: Vec<(f32, f32)> = Vec::new();
        for i in 0..xs.nrows() {
            let e = [xs[[i, 0]], xs[[i, 1]], xs[[i, 2]], xs[[i, 3]]];
            samples.push(decode_phi_psi(e));
        }
        let h1 = rama_hist(&samples, bins);
        jensen_shannon_divergence_histogram(&h_data, &h1, 1e-6)?
    };

    println!();
    println!("PDB 1BPI phi/psi (n={n}) as a torus via R^4 embedding");
    println!("Ramachandran histogram JS divergence (lower is better):");
    println!("- baseline (Gaussian decode): {baseline_js:.4}");
    println!("- trained  (RFM+minibatch OT): {trained_js:.4}");
    println!("- ratio trained/baseline: {:.3}", trained_js / baseline_js);

    Ok(())
}
