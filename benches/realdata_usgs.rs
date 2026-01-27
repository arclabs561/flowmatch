use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flowmatch::metrics::ot_cost_samples_to_weighted_support;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
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

fn load_usgs() -> (Array2<f32>, Array1<f32>) {
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
    let mut y = Array2::<f32>::zeros((n, 3));
    for (i, p) in pts.iter().enumerate() {
        y[[i, 0]] = p[0];
        y[[i, 1]] = p[1];
        y[[i, 2]] = p[2];
    }
    let mut b = Array1::<f32>::zeros(n);
    for i in 0..n {
        b[i] = (mags[i] - 5.0).max(0.0).exp();
    }
    (y, b)
}

fn sample_random_on_sphere(n: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut xs = Array2::<f32>::zeros((n, 3));
    for i in 0..n {
        let v = normalize3([
            StandardNormal.sample(&mut rng),
            StandardNormal.sample(&mut rng),
            StandardNormal.sample(&mut rng),
        ]);
        xs[[i, 0]] = v[0];
        xs[[i, 1]] = v[1];
        xs[[i, 2]] = v[2];
    }
    xs
}

fn bench_realdata_usgs(c: &mut Criterion) {
    let (y, b) = load_usgs();
    let xs = sample_random_on_sphere(512, 999);

    let mut group = c.benchmark_group("flowmatch_realdata_usgs");

    group.bench_function("metric_ot_cost_sinkhorn", |bench| {
        bench.iter(|| {
            let cost = ot_cost_samples_to_weighted_support(
                &black_box(xs.view()),
                &black_box(y.view()),
                &black_box(b.view()),
                0.05,
                800,
                1e-4,
            )
            .unwrap();
            black_box(cost)
        });
    });

    group.bench_function("metric_sliced_wasserstein_tier", |bench| {
        let cfg = DistributionDistanceConfig {
            sw_projections: 32,
            ..Default::default()
        };
        bench.iter(|| {
            let d = DistributionDistance::compute(black_box(y.view()), black_box(xs.view()), &cfg)
                .unwrap();
            black_box(d.sliced_wasserstein.unwrap())
        });
    });

    group.bench_function("train_rfm_minibatch_ot_linear_usgs_short", |bench| {
        let fm_cfg = SdFmTrainConfig {
            lr: 2e-2,
            steps: 400, // intentionally short for benchmarking; scale this up separately.
            batch_size: 64,
            sample_steps: 0,
            seed: 123,
            t_schedule: TimestepSchedule::Uniform,
        };
        let rfm_cfg = RfmMinibatchOtConfig {
            reg: 0.2,
            max_iter: 800,
            tol: 2e-3,
            pairing: RfmMinibatchPairing::SinkhornGreedy,
            pairing_every: 1,
        };
        bench.iter(|| {
            let model = train_rfm_minibatch_ot_linear(
                &black_box(y.view()),
                &black_box(b.view()),
                &rfm_cfg,
                &fm_cfg,
            )
            .unwrap();
            black_box(model)
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_realdata_usgs
}
criterion_main!(benches);
