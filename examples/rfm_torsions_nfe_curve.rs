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

mod common;

use common::mean_std;
use common::torsions::{
    build_torsion_support, decode_phi_psi, parse_phi_psi_csv_2col, rama_hist,
};
use flowmatch::metrics::jensen_shannon_divergence_histogram;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
    TimestepSchedule,
};
use flowmatch::Result;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

fn main() -> Result<()> {
    let phi_psi = parse_phi_psi_csv_2col(20)?;
    let n = phi_psi.len();
    let d = 4usize;
    let (y, b) = build_torsion_support(&phi_psi);

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
        reg: 1.0,
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
