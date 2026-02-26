//! RFM on **real geodata**, evaluated via **cluster-mass matching** (uses `parti`).
//!
//! Uses OT-CFM minibatch coupling (Tong et al., 2023) for straighter flow trajectories.
//!
//! This goes "deeper" than a single OT/JS scalar on raw points by checking whether the model
//! reproduces **mesoscale structure**:
//! - cluster the *real* earthquake points with `parti::Kmeans`,
//! - compute the (magnitude-weighted) **cluster mass distribution**,
//! - assign generated samples to the same centroids and compare the induced cluster-mass
//!   distribution with **Jensen--Shannon divergence** (via `logp` through `flowmatch::metrics`).
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_usgs_earthquakes_cluster_mass
//! ```

mod common;

use common::usgs::{
    argmax_dot_unit, baseline_sphere_samples, build_support_and_weights, normalize3,
    parse_usgs_csv, project_to_sphere,
};
use flowmatch::metrics::jensen_shannon_divergence_histogram;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use flowmatch::Result;
use parti::cluster::{Clustering, Kmeans};
use parti::distribution_distance::{DistributionDistance, DistributionDistanceConfig};

fn main() -> Result<()> {
    let data = parse_usgs_csv(10)?;
    let n = data.pts.len();
    let d = 3usize;
    let (y, b) = build_support_and_weights(&data);

    // ---- "Structure" layer: cluster the real support points.
    let k = 6usize;
    let data_vec: Vec<Vec<f32>> = data.pts.iter().map(|p| vec![p[0], p[1], p[2]]).collect();
    let labels = Kmeans::new(k)
        .with_seed(123)
        .fit_predict(&data_vec)
        .map_err(|_| flowmatch::Error::Domain("parti::Kmeans failed to cluster the USGS points"))?;
    if labels.len() != n {
        return Err(flowmatch::Error::Domain(
            "parti::Kmeans returned wrong label length",
        ));
    }

    // Compute centroids and magnitude-weighted true cluster masses.
    let mut centers = vec![[0.0f32; 3]; k];
    let mut counts = vec![0usize; k];
    let mut true_mass = vec![0.0f32; k];
    let bsum: f32 = b.iter().sum();
    for i in 0..n {
        let c = labels[i];
        if c >= k {
            return Err(flowmatch::Error::Domain(
                "parti::Kmeans produced out-of-range label",
            ));
        }
        centers[c][0] += data.pts[i][0];
        centers[c][1] += data.pts[i][1];
        centers[c][2] += data.pts[i][2];
        counts[c] += 1;
        true_mass[c] += b[i] / bsum;
    }
    for c in 0..k {
        if counts[c] == 0 {
            centers[c] = [1.0, 0.0, 0.0];
        } else {
            centers[c] = normalize3(centers[c]);
        }
    }

    // Baseline: random points on the sphere (projected Gaussian), then induced cluster mass.
    let (baseline_js, baseline_sw) = {
        let m = 2_000usize;
        let xs = baseline_sphere_samples(m, d, 999);
        let mut mass = vec![0.0f32; k];
        for i in 0..m {
            let v = [xs[[i, 0]], xs[[i, 1]], xs[[i, 2]]];
            let c = argmax_dot_unit(v, &centers);
            mass[c] += 1.0 / (m as f32);
        }
        let js = jensen_shannon_divergence_histogram(&true_mass, &mass, 1e-6)?;
        let cfg = DistributionDistanceConfig {
            sw_projections: 32,
            ..Default::default()
        };
        let sw = DistributionDistance::compute(y.view(), xs.view(), &cfg)
            .map_err(|_| flowmatch::Error::Domain("parti::DistributionDistance failed"))?
            .sliced_wasserstein
            .ok_or(flowmatch::Error::Domain(
                "parti sliced_wasserstein unavailable",
            ))?;
        (js, sw)
    };

    // Train + sample.
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
    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;
    let (xs_raw, _js) = model.sample(2_000, 777, fm_cfg.sample_steps)?;

    let (trained_js, trained_sw) = {
        let mut mass = vec![0.0f32; k];
        let mut xs = xs_raw.clone();
        project_to_sphere(&mut xs);
        for i in 0..xs.nrows() {
            let v = [xs[[i, 0]], xs[[i, 1]], xs[[i, 2]]];
            let c = argmax_dot_unit(v, &centers);
            mass[c] += 1.0 / (xs.nrows() as f32);
        }
        let js = jensen_shannon_divergence_histogram(&true_mass, &mass, 1e-6)?;
        let cfg = DistributionDistanceConfig {
            sw_projections: 32,
            ..Default::default()
        };
        let sw = DistributionDistance::compute(y.view(), xs.view(), &cfg)
            .map_err(|_| flowmatch::Error::Domain("parti::DistributionDistance failed"))?
            .sliced_wasserstein
            .ok_or(flowmatch::Error::Domain(
                "parti sliced_wasserstein unavailable",
            ))?;
        (js, sw)
    };

    println!("USGS earthquakes cluster-mass eval (k={k}, n={n})");
    println!("JS(cluster_mass(real), cluster_mass(samples)) (lower is better):");
    println!("- baseline (random on sphere): {baseline_js:.4}");
    println!("- trained  (RFM+minibatch OT): {trained_js:.4}");
    println!("- ratio trained/baseline: {:.3}", trained_js / baseline_js);
    println!("\nSliced Wasserstein (two-sample, lower is better):");
    println!("- baseline (random on sphere): {baseline_sw:.4}");
    println!("- trained  (RFM+minibatch OT): {trained_sw:.4}");
    println!("- ratio trained/baseline: {:.3}", trained_sw / baseline_sw);

    println!("\nReal cluster masses (magnitude-weighted):");
    for c in 0..k {
        println!("  c{c}: mass={:.3}  (count={})", true_mass[c], counts[c]);
    }

    Ok(())
}
