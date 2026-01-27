//! Full pipeline report for the USGS sphere-ish demo.
//!
//! Goal: show the “engine” working end-to-end:
//! - flowmatch training (RFM + minibatch OT)
//! - sampling
//! - multiple evaluation views:
//!   - OT cost to weighted support (wass)
//!   - two-sample sliced Wasserstein (tier + wass)
//!   - cluster-mass JS (tier KMeans + logp)
//!   - graph+community JS (exact kNN + tier Leiden; deterministic)
//!   - optional: kNN via HNSW (tier + jin) + Leiden (non-deterministic build)
//! - timing breakdown per stage
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_usgs_full_pipeline_report
//! ```

use flowmatch::metrics::{
    jensen_shannon_divergence_histogram, ot_cost_samples_to_weighted_support,
};
use flowmatch::sd_fm::RfmMinibatchPairing;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, SdFmTrainConfig};
use flowmatch::{Error, Result};
use ndarray::{Array1, Array2};
use petgraph::graph::UnGraph;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::time::{Duration, Instant};
use tier::cluster::{Clustering, Kmeans};
use tier::community::CommunityDetection;
use tier::distribution_distance::{DistributionDistance, DistributionDistanceConfig};
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

fn cosine_distance_unit(a: [f32; 3], b: [f32; 3]) -> f32 {
    1.0 - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
}

fn exact_knn_graph(points: &[[f32; 3]], k: usize) -> UnGraph<(), f32> {
    let n = points.len();
    let k = k.min(n.saturating_sub(1));
    let mut g = UnGraph::<(), f32>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();

    for i in 0..n {
        let mut dists: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_distance_unit(points[i], points[j])))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(j, dist) in dists.iter().take(k) {
            let w = (1.0 - dist).max(0.001);
            let _ = g.add_edge(nodes[i], nodes[j], w);
        }
    }
    g
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

#[derive(Default)]
struct PhaseTimes {
    load: Duration,
    baseline_sample: Duration,
    train: Duration,
    sample: Duration,
    metrics: Duration,
    exact_knn_leiden: Duration,
    hnsw_knn_leiden: Duration,
}

fn main() -> Result<()> {
    let mut pt = PhaseTimes::default();

    // --- Load USGS
    let t = Instant::now();
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
    pt.load = t.elapsed();

    if pts.len() < 10 {
        return Err(Error::Domain("not enough parsed USGS points"));
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

    // --- Baseline: random on sphere
    let t = Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(999);
    let mut xs0 = Array2::<f32>::zeros((512, d));
    for i in 0..xs0.nrows() {
        let v = normalize3([
            StandardNormal.sample(&mut rng),
            StandardNormal.sample(&mut rng),
            StandardNormal.sample(&mut rng),
        ]);
        xs0[[i, 0]] = v[0];
        xs0[[i, 1]] = v[1];
        xs0[[i, 2]] = v[2];
    }
    pt.baseline_sample = t.elapsed();

    // --- Train model (configurable to support speed/quality tradeoffs)
    let t = Instant::now();
    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: std::env::var("FLOWMATCH_STEPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2_500),
        batch_size: 64,
        sample_steps: 30,
        seed: 123,
        t_schedule: match std::env::var("FLOWMATCH_T_SCHEDULE").as_deref() {
            Ok("ushaped") => TimestepSchedule::UShaped,
            _ => TimestepSchedule::Uniform,
        },
    };
    let keep_frac: f32 = std::env::var("FLOWMATCH_PAIRING_PARTIAL_KEEP_FRAC")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.8);
    let pairing = match std::env::var("FLOWMATCH_PAIRING").as_deref() {
        Ok("rowwise") => RfmMinibatchPairing::RowwiseNearest,
        Ok("exp") => RfmMinibatchPairing::ExpGreedy { temp: 0.2 },
        Ok("partial_rowwise") => RfmMinibatchPairing::PartialRowwise { keep_frac },
        Ok("sinkhorn_selective") => RfmMinibatchPairing::SinkhornSelective { keep_frac },
        _ => RfmMinibatchPairing::SinkhornGreedy,
    };
    let pairing_every = std::env::var("FLOWMATCH_PAIRING_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 0.2,
        max_iter: 2_000,
        tol: 2e-3,
        pairing,
        pairing_every,
    };
    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;
    pt.train = t.elapsed();

    // --- Sample model
    let t = Instant::now();
    let (xs_raw, _js) = model.sample(512, 777, fm_cfg.sample_steps)?;
    let mut xs1 = xs_raw.clone();
    for i in 0..xs1.nrows() {
        let v = normalize3([xs1[[i, 0]], xs1[[i, 1]], xs1[[i, 2]]]);
        xs1[[i, 0]] = v[0];
        xs1[[i, 1]] = v[1];
        xs1[[i, 2]] = v[2];
    }
    pt.sample = t.elapsed();

    // --- Metrics (OT + sliced Wasserstein + cluster-mass JS)
    let t = Instant::now();
    let ot0 =
        ot_cost_samples_to_weighted_support(&xs0.view(), &y.view(), &b.view(), 0.05, 800, 1e-4)?;
    let ot1 =
        ot_cost_samples_to_weighted_support(&xs1.view(), &y.view(), &b.view(), 0.05, 800, 1e-4)?;

    let sw_cfg = DistributionDistanceConfig {
        sw_projections: 32,
        ..Default::default()
    };
    let sw0 = DistributionDistance::compute(y.view(), xs0.view(), &sw_cfg)
        .map_err(|_| Error::Domain("tier::DistributionDistance failed"))?
        .sliced_wasserstein
        .ok_or(Error::Domain("tier sliced_wasserstein unavailable"))?;
    let sw1 = DistributionDistance::compute(y.view(), xs1.view(), &sw_cfg)
        .map_err(|_| Error::Domain("tier::DistributionDistance failed"))?
        .sliced_wasserstein
        .ok_or(Error::Domain("tier sliced_wasserstein unavailable"))?;

    // Cluster mass JS via tier KMeans.
    let k = 6usize;
    let data_vec: Vec<Vec<f32>> = pts.iter().map(|p| vec![p[0], p[1], p[2]]).collect();
    let labels = Kmeans::new(k)
        .with_seed(123)
        .fit_predict(&data_vec)
        .map_err(|_| Error::Domain("tier::Kmeans failed to cluster the USGS points"))?;
    let bsum: f32 = b.sum();
    let mut centers = vec![[0.0f32; 3]; k];
    let mut counts = vec![0usize; k];
    let mut true_mass = vec![0.0f32; k];
    for i in 0..n {
        let c = labels[i];
        centers[c][0] += pts[i][0];
        centers[c][1] += pts[i][1];
        centers[c][2] += pts[i][2];
        counts[c] += 1;
        true_mass[c] += b[i] / bsum;
    }
    for c in 0..k {
        centers[c] = if counts[c] == 0 {
            [1.0, 0.0, 0.0]
        } else {
            normalize3(centers[c])
        };
    }
    let mut mass0 = vec![0.0f32; k];
    let mut mass1 = vec![0.0f32; k];
    for i in 0..xs0.nrows() {
        let v0 = [xs0[[i, 0]], xs0[[i, 1]], xs0[[i, 2]]];
        let v1 = [xs1[[i, 0]], xs1[[i, 1]], xs1[[i, 2]]];
        let c0 = {
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (j, cc) in centers.iter().enumerate() {
                let s = v0[0] * cc[0] + v0[1] * cc[1] + v0[2] * cc[2];
                if s > best_score {
                    best_score = s;
                    best = j;
                }
            }
            best
        };
        let c1 = {
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (j, cc) in centers.iter().enumerate() {
                let s = v1[0] * cc[0] + v1[1] * cc[1] + v1[2] * cc[2];
                if s > best_score {
                    best_score = s;
                    best = j;
                }
            }
            best
        };
        mass0[c0] += 1.0 / (xs0.nrows() as f32);
        mass1[c1] += 1.0 / (xs1.nrows() as f32);
    }
    let js_mass0 = jensen_shannon_divergence_histogram(&true_mass, &mass0, 1e-6)?;
    let js_mass1 = jensen_shannon_divergence_histogram(&true_mass, &mass1, 1e-6)?;

    pt.metrics = t.elapsed();

    // --- Graph+Leiden (deterministic exact kNN)
    let t = Instant::now();
    let leiden = Leiden::new().with_resolution(1.0).with_seed(42);
    let real_graph = exact_knn_graph(&pts, 10.min(n.saturating_sub(1)));
    let labels_real = leiden
        .detect(&real_graph)
        .map_err(|_| Error::Domain("Leiden failed (real exact kNN)"))?;
    let dist_real = community_size_distribution(&labels_real, 16);

    let xs0_vec: Vec<[f32; 3]> = (0..xs0.nrows())
        .map(|i| [xs0[[i, 0]], xs0[[i, 1]], xs0[[i, 2]]])
        .collect();
    let xs1_vec: Vec<[f32; 3]> = (0..xs1.nrows())
        .map(|i| [xs1[[i, 0]], xs1[[i, 1]], xs1[[i, 2]]])
        .collect();

    let g0 = exact_knn_graph(&xs0_vec, 10.min(xs0_vec.len().saturating_sub(1)));
    let g1 = exact_knn_graph(&xs1_vec, 10.min(xs1_vec.len().saturating_sub(1)));
    let lab0 = leiden
        .detect(&g0)
        .map_err(|_| Error::Domain("Leiden failed (baseline exact kNN)"))?;
    let lab1 = leiden
        .detect(&g1)
        .map_err(|_| Error::Domain("Leiden failed (trained exact kNN)"))?;
    let dist0 = community_size_distribution(&lab0, 16);
    let dist1 = community_size_distribution(&lab1, 16);
    let js_leiden0 = jensen_shannon_divergence_histogram(&dist_real, &dist0, 1e-6)?;
    let js_leiden1 = jensen_shannon_divergence_histogram(&dist_real, &dist1, 1e-6)?;
    pt.exact_knn_leiden = t.elapsed();

    // --- Optional: kNN via HNSW (tier + jin); not deterministic, but shows the full stack.
    let t = Instant::now();
    let knn_cfg = KnnGraphConfig {
        k: 10.min(n.saturating_sub(1)),
        symmetric: true,
        weight_fn: WeightFunction::InverseDistance,
        ..Default::default()
    };
    let maybe_hnsw = (|| -> Result<(f32, f32)> {
        let emb_real: Vec<Vec<f32>> = pts.iter().map(|p| vec![p[0], p[1], p[2]]).collect();
        let emb0: Vec<Vec<f32>> = xs0_vec.iter().map(|p| vec![p[0], p[1], p[2]]).collect();
        let emb1: Vec<Vec<f32>> = xs1_vec.iter().map(|p| vec![p[0], p[1], p[2]]).collect();

        let gr = knn_graph_with_config(&emb_real, &knn_cfg)
            .map_err(|_| Error::Domain("tier knn_graph failed (real)"))?;
        let g0 = knn_graph_with_config(&emb0, &knn_cfg)
            .map_err(|_| Error::Domain("tier knn_graph failed (baseline)"))?;
        let g1 = knn_graph_with_config(&emb1, &knn_cfg)
            .map_err(|_| Error::Domain("tier knn_graph failed (trained)"))?;

        let lr = leiden
            .detect(&gr)
            .map_err(|_| Error::Domain("Leiden failed (real HNSW)"))?;
        let l0 = leiden
            .detect(&g0)
            .map_err(|_| Error::Domain("Leiden failed (baseline HNSW)"))?;
        let l1 = leiden
            .detect(&g1)
            .map_err(|_| Error::Domain("Leiden failed (trained HNSW)"))?;

        let dr = community_size_distribution(&lr, 16);
        let d0 = community_size_distribution(&l0, 16);
        let d1 = community_size_distribution(&l1, 16);

        let js0 = jensen_shannon_divergence_histogram(&dr, &d0, 1e-6)?;
        let js1 = jensen_shannon_divergence_histogram(&dr, &d1, 1e-6)?;
        Ok((js0, js1))
    })();
    pt.hnsw_knn_leiden = t.elapsed();

    // --- Report
    println!("USGS full pipeline report (n_support={n}, d={d})");
    println!(
        "train cfg: steps={} pairing={:?} pairing_every={}",
        fm_cfg.steps, rfm_cfg.pairing, rfm_cfg.pairing_every
    );
    println!();
    println!("### Quality metrics (lower is better)");
    println!(
        "- OT cost to weighted support: baseline={ot0:.4}  trained={ot1:.4}  ratio={:.3}",
        ot1 / ot0
    );
    println!(
        "- Sliced Wasserstein (two-sample): baseline={sw0:.4}  trained={sw1:.4}  ratio={:.3}",
        sw1 / sw0
    );
    println!(
        "- Cluster-mass JS (tier KMeans): baseline={js_mass0:.4}  trained={js_mass1:.4}  ratio={:.3}",
        js_mass1 / js_mass0
    );
    println!(
        "- Exact-kNN+Leiden JS (deterministic): baseline={js_leiden0:.4}  trained={js_leiden1:.4}  ratio={:.3}",
        js_leiden1 / js_leiden0
    );
    match maybe_hnsw {
        Ok((js0, js1)) => {
            println!(
                "- HNSW-kNN+Leiden JS (tier+jin; non-deterministic): baseline={js0:.4}  trained={js1:.4}  ratio={:.3}",
                js1 / js0
            );
        }
        Err(_) => {
            println!("- HNSW-kNN+Leiden JS: (skipped due to runtime error)");
        }
    }
    println!();
    println!("### Timing breakdown");
    println!("- load:               {:?}", pt.load);
    println!("- baseline_sample:    {:?}", pt.baseline_sample);
    println!("- train:              {:?}", pt.train);
    println!("- sample:             {:?}", pt.sample);
    println!("- metrics:            {:?}", pt.metrics);
    println!("- exact_knn_leiden:   {:?}", pt.exact_knn_leiden);
    println!("- hnsw_knn_leiden:    {:?}", pt.hnsw_knn_leiden);

    Ok(())
}
