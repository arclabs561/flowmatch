use flowmatch::metrics::jensen_shannon_divergence_histogram;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use flowmatch::Result;
use ndarray::{Array1, Array2};
use petgraph::graph::UnGraph;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use tier::community::CommunityDetection;
use tier::Leiden;

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
    // For unit vectors: cosine distance = 1 - dot in [0,2].
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
            let w = (1.0 - dist).max(0.001); // similarity-like weight
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

#[test]
fn rfm_usgs_improves_leiden_structure_js_under_exact_knn() -> Result<()> {
    // Load support points.
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
    assert!(n >= 10);

    // Real communities on exact kNN graph.
    let k = 10usize.min(n.saturating_sub(1));
    let g_real = exact_knn_graph(&pts, k);
    let leiden = Leiden::new().with_resolution(1.0).with_seed(42);
    let labels_real = leiden.detect(&g_real).unwrap();
    let dist_real = community_size_distribution(&labels_real, 16);

    // Baseline: average over multiple random seeds (reduces “got lucky” cases).
    let baseline_js_mean = {
        let mut vals: Vec<f32> = Vec::new();
        for seed in [100u64, 200, 300, 400, 500] {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut xs: Vec<[f32; 3]> = Vec::new();
            for _ in 0..250 {
                xs.push(normalize3([
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                ]));
            }
            let g = exact_knn_graph(&xs, 10.min(xs.len().saturating_sub(1)));
            let labels = leiden.detect(&g).unwrap();
            let dist = community_size_distribution(&labels, 16);
            vals.push(jensen_shannon_divergence_histogram(
                &dist_real, &dist, 1e-6,
            )?);
        }
        vals.iter().sum::<f32>() / (vals.len() as f32)
    };

    // Train + sample.
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
    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: 2_000,
        batch_size: 64,
        sample_steps: 25,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 0.2,
        max_iter: 1_500,
        tol: 2e-3,
        // New coupling discipline: avoid forcing rare/outlier targets inside minibatches.
        // This keeps Sinkhorn's global signal but relaxes the "use every column" constraint
        // in the extracted assignment.
        pairing: RfmMinibatchPairing::SinkhornSelective { keep_frac: 0.8 },
        pairing_every: 2,
    };
    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;
    // Trained score is also averaged across multiple sampling seeds to reduce flakiness
    // (the graph/community pipeline is sensitive to small sample differences).
    let trained_js_mean = {
        let seeds = [777u64, 778, 779, 780, 781];
        let mut sum = 0.0f64;
        for &seed in &seeds {
            let (xs_raw, _js) = model.sample(300, seed, fm_cfg.sample_steps)?;
            let mut xs: Vec<[f32; 3]> = Vec::new();
            for i in 0..xs_raw.nrows() {
                xs.push(normalize3([xs_raw[[i, 0]], xs_raw[[i, 1]], xs_raw[[i, 2]]]));
            }
            let g = exact_knn_graph(&xs, 10.min(xs.len().saturating_sub(1)));
            let labels = leiden.detect(&g).unwrap();
            let dist = community_size_distribution(&labels, 16);
            let js = jensen_shannon_divergence_histogram(&dist_real, &dist, 1e-6)? as f64;
            sum += js;
        }
        (sum / seeds.len() as f64) as f32
    };

    assert!(
        trained_js_mean < baseline_js_mean,
        "expected trained_js_mean < baseline_js_mean; trained_mean={trained_js_mean} baseline_mean={baseline_js_mean}"
    );

    Ok(())
}
