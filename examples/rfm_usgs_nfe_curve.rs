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

mod common;

use common::mean_std;
use common::usgs::{
    baseline_sphere_samples, build_support_and_weights, normalize3, parse_usgs_csv,
};
use flowmatch::metrics::ot_cost_samples_to_weighted_support;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
    TimestepSchedule,
};
use flowmatch::Result;
use ndarray::Array2;

fn main() -> Result<()> {
    let data = parse_usgs_csv(10)?;
    let n = data.pts.len();
    let d = 3usize;
    let (y, b) = build_support_and_weights(&data);

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
        reg: 1.0,
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
        let xs0 = baseline_sphere_samples(m, d, seed);
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
