//! USGS RFM: Euler vs Heun under equal NFE budgets.
//!
//! Papers report quality vs NFE (number of function evaluations), not "steps".
//! Heun uses 2 velocity evaluations per step, so compare:
//! - Euler with `nfe` steps
//! - Heun with `nfe/2` steps (same NFE budget)
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_usgs_solver_nfe_tradeoff
//! ```
//!
//! Optional:
//! ```bash
//! FLOWMATCH_T_SCHEDULE=ushaped cargo run -p flowmatch --example rfm_usgs_solver_nfe_tradeoff
//! ```

mod common;

use common::mean_std;
use common::usgs::{
    baseline_sphere_samples, build_support_and_weights, normalize3, parse_usgs_csv,
};
use flowmatch::metrics::ot_cost_samples_to_weighted_support;
use flowmatch::ode::OdeMethod;
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
    let b_norm = &b / b.sum();

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

    // Baseline: Gaussian -> project to sphere.
    let m = 1_500usize;
    let seeds = [900u64, 901, 902, 903, 904];
    let mut baseline_vals: Vec<f64> = Vec::new();
    for &seed in &seeds {
        let xs0 = baseline_sphere_samples(m, d, seed);
        let ot = ot_cost_samples_to_weighted_support(
            &xs0.view(),
            &y.view(),
            &b_norm.view(),
            0.10,
            8_000,
            2e-3,
        )?;
        baseline_vals.push(ot as f64);
    }
    let (baseline_mean, baseline_std) = mean_std(&baseline_vals);

    println!("USGS solver NFE tradeoff (n_support={n}, d={d})");
    println!("train t_schedule={t_schedule:?}");
    println!("baseline OT cost: mean={baseline_mean:.4}  std={baseline_std:.4}");
    println!();

    for nfe in [2usize, 4, 8, 16, 32] {
        // Euler: steps = NFE
        let mut e_vals: Vec<f64> = Vec::new();
        // Heun: 2 evals per step => steps = NFE/2 (rounded down, min 1)
        let heun_steps = (nfe / 2).max(1);
        let mut h_vals: Vec<f64> = Vec::new();

        for &seed in &seeds {
            // Euler
            let (_x0s, xs_raw, _js) =
                model.sample_with_x0_method(m, seed, nfe, OdeMethod::Euler)?;
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
                &b_norm.view(),
                0.10,
                8_000,
                2e-3,
            )?;
            e_vals.push(ot as f64);

            // Heun (equal NFE budget: 2*heun_steps <= nfe)
            let (_x0s2, xs_raw2, _js2) =
                model.sample_with_x0_method(m, seed, heun_steps, OdeMethod::Heun)?;
            let mut xs2 = Array2::<f32>::zeros((m, d));
            for i in 0..m {
                let v = normalize3([xs_raw2[[i, 0]], xs_raw2[[i, 1]], xs_raw2[[i, 2]]]);
                xs2[[i, 0]] = v[0];
                xs2[[i, 1]] = v[1];
                xs2[[i, 2]] = v[2];
            }
            let ot2 = ot_cost_samples_to_weighted_support(
                &xs2.view(),
                &y.view(),
                &b_norm.view(),
                0.10,
                8_000,
                2e-3,
            )?;
            h_vals.push(ot2 as f64);
        }

        let (em, es) = mean_std(&e_vals);
        let (hm, hs) = mean_std(&h_vals);
        println!(
            "NFE={:>2}  Euler(steps={:>2}): mean={:.4}+/-{:.4} | Heun(steps={:>2}, NFE~{}): mean={:.4}+/-{:.4}",
            nfe,
            nfe,
            em,
            es,
            heun_steps,
            2 * heun_steps,
            hm,
            hs
        );
    }

    Ok(())
}
