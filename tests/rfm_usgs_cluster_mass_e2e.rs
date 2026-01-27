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

#[test]
fn rfm_usgs_cluster_mass_js_improves_vs_random_sphere() -> Result<()> {
    // Parse.
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

    // y + weights.
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

    // Cluster real points.
    let k = 6usize;
    let data_vec: Vec<Vec<f32>> = pts.iter().map(|p| vec![p[0], p[1], p[2]]).collect();
    let labels = Kmeans::new(k)
        .with_seed(123)
        .fit_predict(&data_vec)
        .unwrap();
    assert_eq!(labels.len(), n);

    let mut centers = vec![[0.0f32; 3]; k];
    let mut counts = vec![0usize; k];
    let mut true_mass = vec![0.0f32; k];
    let bsum: f32 = b.iter().sum();
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

    // Baseline: average multiple random-on-sphere draws to reduce flakiness.
    let (baseline_js_mean, baseline_sw_mean) = {
        let cfg = DistributionDistanceConfig {
            sw_projections: 16,
            ..Default::default()
        };
        let m = 1_500usize;
        let seeds = [900u64, 901, 902, 903, 904];
        let mut js_sum = 0.0f64;
        let mut sw_sum = 0.0f64;

        for &seed in &seeds {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut mass = vec![0.0f32; k];
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
            let js = jensen_shannon_divergence_histogram(&true_mass, &mass, 1e-6)? as f64;
            let sw = DistributionDistance::compute(y.view(), xs.view(), &cfg)
                .unwrap()
                .sliced_wasserstein
                .unwrap() as f64;
            js_sum += js;
            sw_sum += sw;
        }

        let denom = seeds.len() as f64;
        ((js_sum / denom) as f32, (sw_sum / denom) as f32)
    };

    // Train + evaluate JS on samples.
    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: 2_000, // keep tests fast
        batch_size: 64,
        sample_steps: 25,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 0.2,
        max_iter: 1_500,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 4,
    };
    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;
    let (xs_raw, _js) = model.sample(1_500, 777, fm_cfg.sample_steps)?;

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
            sw_projections: 16,
            ..Default::default()
        };
        let sw = DistributionDistance::compute(y.view(), xs.view(), &cfg)
            .unwrap()
            .sliced_wasserstein
            .unwrap();
        (js, sw)
    };

    // This is a structural check: learned flow should beat random-on-sphere in cluster mass.
    assert!(
        trained_js < baseline_js_mean,
        "expected trained_js < baseline_js_mean; trained={trained_js} baseline_mean={baseline_js_mean}"
    );
    assert!(
        trained_sw < baseline_sw_mean,
        "expected trained_sw < baseline_sw_mean; trained={trained_sw} baseline_mean={baseline_sw_mean}"
    );

    Ok(())
}
