//! RFM on **real geodata**, evaluated via **cluster-mass matching** (uses `tier`).
//!
//! This goes “deeper” than a single OT/JS scalar on raw points by checking whether the model
//! reproduces **mesoscale structure**:
//! - cluster the *real* earthquake points with `tier::Kmeans`,
//! - compute the (magnitude-weighted) **cluster mass distribution**,
//! - assign generated samples to the same centroids and compare the induced cluster-mass
//!   distribution with **Jensen–Shannon divergence** (via `logp` through `flowmatch::metrics`).
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_usgs_earthquakes_cluster_mass
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
use tier::cluster::{Clustering, Kmeans};
use tier::distribution_distance::{DistributionDistance, DistributionDistanceConfig};

const USGS_CSV: &str = include_str!("../examples_data/usgs_eq_m6_2024_limit50.csv.txt");

fn deg_to_rad(x: f32) -> f32 {
    x * core::f32::consts::PI / 180.0
}

fn latlon_to_unit_xyz(lat_deg: f32, lon_deg: f32) -> [f32; 3] {
    let lat = deg_to_rad(lat_deg);
    let lon = deg_to_rad(lon_deg);
    let clat = lat.cos();
    [clat * lon.cos(), clat * lon.sin(), lat.sin()]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 0.0 && n.is_finite() {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn argmax_dot_unit(v: [f32; 3], centers: &[[f32; 3]]) -> usize {
    // centers are unit vectors; dot is cosine similarity.
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, c) in centers.iter().enumerate() {
        let s = v[0] * c[0] + v[1] * c[1] + v[2] * c[2];
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    best
}

fn main() -> Result<()> {
    // Parse points + magnitudes from the vendored USGS CSV.
    let mut pts: Vec<[f32; 3]> = Vec::new();
    let mut mags: Vec<f32> = Vec::new();
    for (line_idx, line) in USGS_CSV.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        let lat: f32 = parts[1].parse().unwrap_or(0.0);
        let lon: f32 = parts[2].parse().unwrap_or(0.0);
        let mag: f32 = parts[4].parse().unwrap_or(0.0);
        if !lat.is_finite() || !lon.is_finite() || !mag.is_finite() {
            continue;
        }
        pts.push(latlon_to_unit_xyz(lat, lon));
        mags.push(mag);
    }
    if pts.len() < 10 {
        return Err(flowmatch::Error::Domain("not enough parsed USGS points"));
    }

    let n = pts.len();
    let d = 3usize;

    // Discrete support y.
    let mut y = Array2::<f32>::zeros((n, d));
    for (i, p) in pts.iter().enumerate() {
        y[[i, 0]] = p[0];
        y[[i, 1]] = p[1];
        y[[i, 2]] = p[2];
    }

    // Magnitude-derived weights (explicitly normalized inside training and metrics).
    let mut b = Array1::<f32>::zeros(n);
    for i in 0..n {
        b[i] = (mags[i] - 5.0).max(0.0).exp();
    }

    // ---- “Structure” layer: cluster the real support points.
    // Kmeans operates in Euclidean space; for unit vectors this is fine.
    let k = 6usize;
    let data_vec: Vec<Vec<f32>> = pts.iter().map(|p| vec![p[0], p[1], p[2]]).collect();
    let labels = Kmeans::new(k)
        .with_seed(123)
        .fit_predict(&data_vec)
        .map_err(|_| flowmatch::Error::Domain("tier::Kmeans failed to cluster the USGS points"))?;
    if labels.len() != n {
        return Err(flowmatch::Error::Domain(
            "tier::Kmeans returned wrong label length",
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
                "tier::Kmeans produced out-of-range label",
            ));
        }
        centers[c][0] += pts[i][0];
        centers[c][1] += pts[i][1];
        centers[c][2] += pts[i][2];
        counts[c] += 1;
        true_mass[c] += b[i] / bsum;
    }
    for c in 0..k {
        if counts[c] == 0 {
            // Rare but possible; keep a placeholder center.
            centers[c] = [1.0, 0.0, 0.0];
        } else {
            centers[c] = normalize3(centers[c]);
        }
    }

    // Baseline: random points on the sphere (projected Gaussian), then induced cluster mass.
    let (baseline_js, baseline_sw) = {
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let mut mass = vec![0.0f32; k];
        let m = 2_000usize;
        let mut xs = Array2::<f32>::zeros((m, d));
        for i in 0..m {
            let v = normalize3([
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ]);
            let c = argmax_dot_unit(v, &centers);
            mass[c] += 1.0 / (m as f32);
            xs[[i, 0]] = v[0];
            xs[[i, 1]] = v[1];
            xs[[i, 2]] = v[2];
        }
        let js = jensen_shannon_divergence_histogram(&true_mass, &mass, 1e-6)?;
        let cfg = DistributionDistanceConfig {
            sw_projections: 32,
            ..Default::default()
        };
        let sw = DistributionDistance::compute(y.view(), xs.view(), &cfg)
            .map_err(|_| flowmatch::Error::Domain("tier::DistributionDistance failed"))?
            .sliced_wasserstein
            .ok_or(flowmatch::Error::Domain(
                "tier sliced_wasserstein unavailable",
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
        reg: 0.2,
        max_iter: 2_000,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 1,
    };
    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;
    let (xs_raw, _js) = model.sample(2_000, 777, fm_cfg.sample_steps)?;

    let (trained_js, trained_sw) = {
        let mut mass = vec![0.0f32; k];
        let mut xs = Array2::<f32>::zeros((xs_raw.nrows(), d));
        for i in 0..xs_raw.nrows() {
            let v = normalize3([xs_raw[[i, 0]], xs_raw[[i, 1]], xs_raw[[i, 2]]]);
            let c = argmax_dot_unit(v, &centers);
            mass[c] += 1.0 / (xs_raw.nrows() as f32);
            xs[[i, 0]] = v[0];
            xs[[i, 1]] = v[1];
            xs[[i, 2]] = v[2];
        }
        let js = jensen_shannon_divergence_histogram(&true_mass, &mass, 1e-6)?;
        let cfg = DistributionDistanceConfig {
            sw_projections: 32,
            ..Default::default()
        };
        let sw = DistributionDistance::compute(y.view(), xs.view(), &cfg)
            .map_err(|_| flowmatch::Error::Domain("tier::DistributionDistance failed"))?
            .sliced_wasserstein
            .ok_or(flowmatch::Error::Domain(
                "tier sliced_wasserstein unavailable",
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
