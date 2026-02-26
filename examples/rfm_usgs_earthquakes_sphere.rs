//! Rectified flow matching on **real geospatial data** (earthquakes on the sphere).
//!
//! This example is motivated by "Flow Matching on General Geometries" style benchmarks where
//! *sphere-valued* data (lat/lon) is the native object. We stay within `flowmatch`'s current
//! constraints (a tiny linear conditional field in Euclidean space) but:
//!
//! - we use a **real dataset** (USGS earthquake catalog),
//! - we evaluate with a **distributional metric** (entropic OT cost to a weighted support),
//! - we keep determinism explicit (seeded).
//!
//! Data provenance:
//! - Source: USGS Earthquake Catalog API (CSV)
//! - Query used (limit 50, M≥6, year 2024):
//!   `https://earthquake.usgs.gov/fdsnws/event/1/query.csv?format=csv&starttime=2024-01-01&endtime=2024-12-31&minmagnitude=6&orderby=time&limit=50`
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_usgs_earthquakes_sphere
//! ```

mod common;

use common::usgs::{
    baseline_sphere_samples, build_support_and_weights, parse_usgs_csv, project_to_sphere,
    unit_xyz_to_latlon,
};
use flowmatch::metrics::ot_cost_samples_to_weighted_support;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
    TimestepSchedule,
};
use flowmatch::Result;

fn main() -> Result<()> {
    let data = parse_usgs_csv(10)?;
    let n = data.pts.len();
    let d = 3usize;
    let (y, b) = build_support_and_weights(&data);

    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: 2_500,
        batch_size: 64,
        sample_steps: 30,
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

    // Baseline: raw Gaussian noise (no learned flow), then project to the sphere.
    let xs0 = baseline_sphere_samples(256, d, 999);
    let baseline =
        ot_cost_samples_to_weighted_support(&xs0.view(), &y.view(), &b.view(), 0.05, 800, 1e-4)?;

    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;

    let (xs_raw, _js) = model.sample(256, 777, fm_cfg.sample_steps)?;
    let mut xs = xs_raw.clone();
    project_to_sphere(&mut xs);

    let trained =
        ot_cost_samples_to_weighted_support(&xs.view(), &y.view(), &b.view(), 0.05, 800, 1e-4)?;

    println!("USGS earthquakes (n={n}), embedding=R^3 with S^2 projection");
    println!("OT cost (lower is better):");
    println!("- baseline (near-noise): {baseline:.4}");
    println!("- trained  (RFM+minibatch OT): {trained:.4}");
    println!("- ratio trained/baseline: {:.3}", trained / baseline);

    println!("\nSome generated samples (lat, lon):");
    for i in 0..8.min(xs.nrows()) {
        let (lat, lon) = unit_xyz_to_latlon([xs[[i, 0]], xs[[i, 1]], xs[[i, 2]]]);
        println!("  {i:>2}: lat={lat:>7.2}, lon={lon:>8.2}");
    }

    Ok(())
}
