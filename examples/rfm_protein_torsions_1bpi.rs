//! Rectified flow matching on **real protein torsion data** (a torus-shaped domain).
//!
//! Motivation (from “Flow Matching on General Geometries” and FoldFlow-style eval culture):
//! torsion angles live on a product of circles \(S^1 \times S^1\) (a 2D torus), and *distribution*
//! matching should be measurable (not just “the code runs”).
//!
//! We stay within `flowmatch`’s current primitive (linear conditional field + minibatch OT pairing),
//! but we use **real φ/ψ** angles extracted from a PDB structure and score matching via a simple,
//! interpretable distributional metric: **JS divergence** between Ramachandran histograms
//! (computed via the ecosystem `logp` crate through `flowmatch::metrics`).
//!
//! Data provenance:
//! - PDB: `1BPI` chain A (BPTI). Source: RCSB PDB (`https://files.rcsb.org/download/1BPI.pdb`)
//! - This repo vendors a tiny derived artifact: φ/ψ angles (radians) computed from backbone atoms.
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_protein_torsions_1bpi
//! ```

use flowmatch::metrics::jensen_shannon_divergence_histogram;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use flowmatch::Result;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

const PHI_PSI_CSV: &str = include_str!("../examples_data/pdb_1bpi_phi_psi.csv.txt");

fn wrap_pi(x: f32) -> f32 {
    // Map to (-pi, pi]
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
        let u = (wrap_pi(phi) + core::f32::consts::PI) / two_pi; // [0,1)
        let v = (wrap_pi(psi) + core::f32::consts::PI) / two_pi; // [0,1)
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

fn main() -> Result<()> {
    // Parse φ/ψ pairs (radians).
    let mut phi_psi: Vec<(f32, f32)> = Vec::new();
    for (line_idx, line) in PHI_PSI_CSV.lines().enumerate() {
        if line_idx == 0 {
            continue;
        }
        if line.trim().is_empty() {
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

    if phi_psi.len() < 20 {
        return Err(flowmatch::Error::Domain("not enough parsed phi/psi pairs"));
    }

    // Discrete support: embeddings of φ/ψ on (S^1)^2 ⊂ R^4.
    let n = phi_psi.len();
    let d = 4usize;
    let mut y = Array2::<f32>::zeros((n, d));
    for (i, (phi, psi)) in phi_psi.iter().copied().enumerate() {
        let e = embed_phi_psi(phi, psi);
        for k in 0..d {
            y[[i, k]] = e[k];
        }
    }
    let b = Array1::<f32>::from_elem(n, 1.0); // uniform weights

    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: 3_000,
        batch_size: 64,
        sample_steps: 40,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 0.2,
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
            // Interpret four independent Gaussians as (cos,sin,cos,sin) and decode.
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

    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;
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

    println!("PDB 1BPI φ/ψ (n={n}) as a torus via R^4 embedding");
    println!("Ramachandran histogram JS divergence (lower is better):");
    println!("- baseline (Gaussian decode): {baseline_js:.4}");
    println!("- trained  (RFM+minibatch OT): {trained_js:.4}");
    println!("- ratio trained/baseline: {:.3}", trained_js / baseline_js);

    Ok(())
}
