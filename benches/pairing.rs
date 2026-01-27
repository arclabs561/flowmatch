use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use flowmatch::rfm::{
    minibatch_exp_greedy_pairing, minibatch_ot_greedy_pairing, minibatch_ot_selective_pairing,
    minibatch_partial_rowwise_pairing, minibatch_rowwise_nearest_pairing,
};

fn make_xy(n: usize, d: usize, seed: u64) -> (Array2<f32>, Array2<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x = Array2::<f32>::zeros((n, d));
    let mut y = Array2::<f32>::zeros((n, d));
    for i in 0..n {
        for k in 0..d {
            x[[i, k]] = StandardNormal.sample(&mut rng);
            y[[i, k]] = StandardNormal.sample(&mut rng);
        }
    }
    (x, y)
}

fn bench_pairing(c: &mut Criterion) {
    let mut group = c.benchmark_group("rfm_pairing");
    group.sample_size(30);

    let cases = [
        (32usize, 3usize),
        (64usize, 3usize),
        (64usize, 8usize),
        (128usize, 3usize),
    ];

    for &(n, d) in &cases {
        let (x, y) = make_xy(n, d, 123);

        group.bench_with_input(
            BenchmarkId::new("rowwise_nearest", format!("n{n}_d{d}")),
            &(n, d),
            |b, _| b.iter(|| minibatch_rowwise_nearest_pairing(&x.view(), &y.view()).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("partial_rowwise", format!("n{n}_d{d}")),
            &(n, d),
            |b, _| b.iter(|| minibatch_partial_rowwise_pairing(&x.view(), &y.view(), 0.8).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("exp_greedy", format!("n{n}_d{d}")),
            &(n, d),
            |b, _| b.iter(|| minibatch_exp_greedy_pairing(&x.view(), &y.view(), 0.2).unwrap()),
        );

        // Keep Sinkhorn cheap-ish for benches; the goal is comparative scaling, not convergence bragging.
        group.bench_with_input(
            BenchmarkId::new("sinkhorn_greedy", format!("n{n}_d{d}")),
            &(n, d),
            |b, _| {
                b.iter(|| {
                    minibatch_ot_greedy_pairing(&x.view(), &y.view(), 0.2, 300, 2e-3).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sinkhorn_selective", format!("n{n}_d{d}")),
            &(n, d),
            |b, _| {
                b.iter(|| {
                    minibatch_ot_selective_pairing(&x.view(), &y.view(), 0.2, 300, 2e-3, 0.8)
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_pairing);
criterion_main!(benches);
