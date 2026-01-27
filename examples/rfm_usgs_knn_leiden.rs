//! Full-pipeline structure demo (flowmatch + wass + tier + jin):
//!
//! - Train RFM on real USGS earthquake locations (sphere in R^3).
//! - Sample points from the learned flow.
//! - Build a kNN graph using `tier`'s `knn_graph_with_config` (HNSW via `jin`).
//! - Run Leiden community detection and compare **community-size distributions**
//!   against the real-data graph via JS divergence.
//!
//! This is meant as a “how the engine composes” example, not a perfectly deterministic test:
//! HNSW construction involves randomness internally.
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_usgs_knn_leiden
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
use tier::community::CommunityDetection;
use tier::{knn_graph_with_config, KnnGraphConfig, Leiden, WeightFunction};

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

fn community_size_distribution(labels: &[usize], top_k: usize) -> Vec<f32> {
    use std::collections::HashMap;
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &c in labels {
        *counts.entry(c).or_insert(0) += 1;
    }
    let mut sizes: Vec<f32> = counts.values().map(|&x| x as f32).collect();
    sizes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = sizes.iter().sum();
    if total > 0.0 {
        for x in &mut sizes {
            *x /= total;
        }
    }
    sizes.truncate(top_k);
    while sizes.len() < top_k {
        sizes.push(0.0);
    }
    sizes
}

fn main() -> Result<()> {
    // Load support + weights.
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

    let n = pts.len();
    let d = 3usize;
    let mut y = Array2::<f32>::zeros((n, d));
    for (i, p) in pts.iter().enumerate() {
        y[[i, 0]] = p[0];
        y[[i, 1]] = p[1];
        y[[i, 2]] = p[2];
    }
    let mut b = Array1::<f32>::zeros(n);
    for i in 0..n {
        b[i] = (mags[i] - 5.0).max(0.0).exp();
    }

    // Build real kNN graph + Leiden labels.
    let embeddings_real: Vec<Vec<f32>> = pts.iter().map(|p| vec![p[0], p[1], p[2]]).collect();
    let knn_cfg = KnnGraphConfig {
        k: 10.min(n.saturating_sub(1)),
        symmetric: true,
        weight_fn: WeightFunction::InverseDistance,
        ..Default::default()
    };
    let graph_real = knn_graph_with_config(&embeddings_real, &knn_cfg)
        .map_err(|_| flowmatch::Error::Domain("tier::knn_graph_with_config failed"))?;
    let leiden = Leiden::new().with_resolution(1.0).with_seed(42);
    let labels_real = leiden
        .detect(&graph_real)
        .map_err(|_| flowmatch::Error::Domain("Leiden failed on real graph"))?;
    let real_sizes = community_size_distribution(&labels_real, 16);

    // Baseline: random-on-sphere samples (no learning), then kNN+Leiden.
    let baseline_js = {
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let mut emb: Vec<Vec<f32>> = Vec::new();
        for _ in 0..400 {
            let v = normalize3([
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ]);
            emb.push(vec![v[0], v[1], v[2]]);
        }
        let g = knn_graph_with_config(&emb, &knn_cfg)
            .map_err(|_| flowmatch::Error::Domain("tier knn_graph failed (baseline)"))?;
        let lab = leiden
            .detect(&g)
            .map_err(|_| flowmatch::Error::Domain("Leiden failed (baseline)"))?;
        let sizes = community_size_distribution(&lab, 16);
        jensen_shannon_divergence_histogram(&real_sizes, &sizes, 1e-6)?
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
    let (xs_raw, _js) = model.sample(400, 777, fm_cfg.sample_steps)?;

    let trained_js = {
        let mut emb: Vec<Vec<f32>> = Vec::new();
        for i in 0..xs_raw.nrows() {
            let v = normalize3([xs_raw[[i, 0]], xs_raw[[i, 1]], xs_raw[[i, 2]]]);
            emb.push(vec![v[0], v[1], v[2]]);
        }
        let g = knn_graph_with_config(&emb, &knn_cfg)
            .map_err(|_| flowmatch::Error::Domain("tier knn_graph failed (trained)"))?;
        let lab = leiden
            .detect(&g)
            .map_err(|_| flowmatch::Error::Domain("Leiden failed (trained)"))?;
        let sizes = community_size_distribution(&lab, 16);
        jensen_shannon_divergence_histogram(&real_sizes, &sizes, 1e-6)?
    };

    println!("USGS kNN+Leiden structure eval (non-deterministic HNSW build)");
    println!("JS(community-size(real), community-size(samples)) lower is better:");
    println!("- baseline: {baseline_js:.4}");
    println!("- trained : {trained_js:.4}");
    println!("- ratio  : {:.3}", trained_js / baseline_js);

    Ok(())
}
