use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flowmatch::metrics::jensen_shannon_divergence_histogram;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

const PHI_PSI_CSV: &str = include_str!("../examples_data/pdb_1bpi_phi_psi.csv.txt");

fn wrap_pi(x: f32) -> f32 {
    let mut y = x % (2.0 * core::f32::consts::PI);
    if y <= -core::f32::consts::PI {
        y += 2.0 * core::f32::consts::PI;
    }
    if y > core::f32::consts::PI {
        y -= 2.0 * core::f32::consts::PI;
    }
    y
}

fn embed_phi_psi(phi: f32, psi: f32) -> [f32; 4] {
    [phi.cos(), phi.sin(), psi.cos(), psi.sin()]
}

fn decode_phi_psi(e: [f32; 4]) -> (f32, f32) {
    let phi = e[1].atan2(e[0]);
    let psi = e[3].atan2(e[2]);
    (wrap_pi(phi), wrap_pi(psi))
}

fn rama_hist(phi_psi: &[(f32, f32)], bins: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; bins * bins];
    let two_pi = 2.0 * core::f32::consts::PI;
    for &(phi, psi) in phi_psi {
        let u = (wrap_pi(phi) + core::f32::consts::PI) / two_pi;
        let v = (wrap_pi(psi) + core::f32::consts::PI) / two_pi;
        let mut i = (u * bins as f32).floor() as isize;
        let mut j = (v * bins as f32).floor() as isize;
        if i < 0 {
            i = 0;
        }
        if j < 0 {
            j = 0;
        }
        if i >= bins as isize {
            i = bins as isize - 1;
        }
        if j >= bins as isize {
            j = bins as isize - 1;
        }
        h[(i as usize) * bins + (j as usize)] += 1.0;
    }
    let sum: f32 = h.iter().sum();
    if sum > 0.0 {
        for x in &mut h {
            *x /= sum;
        }
    }
    h
}

fn load_phi_psi() -> (Vec<(f32, f32)>, Array2<f32>, Array1<f32>) {
    let mut phi_psi: Vec<(f32, f32)> = Vec::new();
    for (line_idx, line) in PHI_PSI_CSV.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        let phi: f32 = parts[4].parse().unwrap_or(0.0);
        let psi: f32 = parts[5].parse().unwrap_or(0.0);
        if phi.is_finite() && psi.is_finite() {
            phi_psi.push((phi, psi));
        }
    }

    let n = phi_psi.len();
    let mut y = Array2::<f32>::zeros((n, 4));
    for (i, (phi, psi)) in phi_psi.iter().copied().enumerate() {
        let e = embed_phi_psi(phi, psi);
        for k in 0..4 {
            y[[i, k]] = e[k];
        }
    }
    let b = Array1::<f32>::from_elem(n, 1.0);
    (phi_psi, y, b)
}

fn bench_realdata_torsions(c: &mut Criterion) {
    let (phi_psi, y, b) = load_phi_psi();
    let bins = 36usize;
    let h_data = rama_hist(&phi_psi, bins);

    let mut group = c.benchmark_group("flowmatch_realdata_torsions");

    group.bench_function("metric_ramachandran_js_baseline_gaussian", |bench| {
        bench.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(999);
            let mut samples: Vec<(f32, f32)> = Vec::new();
            for _ in 0..512 {
                let e = [
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                    StandardNormal.sample(&mut rng),
                ];
                samples.push(decode_phi_psi(e));
            }
            let h0 = rama_hist(&samples, bins);
            let js = jensen_shannon_divergence_histogram(&h_data, &h0, 1e-6).unwrap();
            black_box(js)
        });
    });

    group.bench_function("train_rfm_minibatch_ot_linear_torsions_short", |bench| {
        let fm_cfg = SdFmTrainConfig {
            lr: 2e-2,
            steps: 400,
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
    targets = bench_realdata_torsions
}
criterion_main!(benches);
