//! Integration test: Fisher Flow Matching is the generic Riemannian FM loop
//! (`train_riemannian_fm`, Chen & Lipman 2023) instantiated on `infogeom`'s
//! `FisherRaoSimplex` (Davis et al. 2024). This is the CI-enforced proof that
//! the cross-crate composition actually drives the integrator: the worked
//! example (`examples/rfm_fisher_simplex.rs`) only compiles under `cargo test`,
//! its assertions never run. This test runs them.
//!
//! Minimal by design: a single categorical mode and a linear tangent field,
//! which is enough to show training transports mass toward the mode along the
//! Fisher-Rao geodesics while keeping every point on the simplex. The richer
//! multi-modal MLP demonstration lives in the example.

#![cfg(feature = "riemannian")]

use flowmatch::ode::OdeMethod;
use flowmatch::riemannian::{train_riemannian_fm, ManifoldVectorField};
use flowmatch::riemannian_ode::integrate_fixed_manifold;
use flowmatch::sd_fm::SdFmTrainConfig;
use infogeom::FisherRaoSimplex;
use ndarray::{Array1, Array2, ArrayView1};

const DIM: usize = 3;

/// A linear tangent field `v(x, t) = W * [x; t] + b`. A single mode is
/// linearly transportable, so this minimal field suffices for the test.
struct LinearTangentField {
    w: Array2<f32>, // DIM x (DIM + 1)
    b: Array1<f32>, // DIM
}

impl LinearTangentField {
    fn new() -> Self {
        Self {
            w: Array2::zeros((DIM, DIM + 1)),
            b: Array1::zeros(DIM),
        }
    }

    fn input(x: &ArrayView1<f32>, t: f32) -> Array1<f32> {
        let mut inp = Array1::<f32>::zeros(DIM + 1);
        for j in 0..DIM {
            inp[j] = x[j];
        }
        inp[DIM] = t;
        inp
    }
}

impl ManifoldVectorField for LinearTangentField {
    fn eval(&self, x: &ArrayView1<f32>, t: f32) -> Array1<f32> {
        self.w.dot(&Self::input(x, t)) + &self.b
    }

    fn sgd_step(&mut self, x: &ArrayView1<f32>, t: f32, target: &ArrayView1<f32>, lr: f32) {
        let inp = Self::input(x, t);
        let d_out = &self.eval(x, t) - target;
        for i in 0..DIM {
            for j in 0..DIM + 1 {
                self.w[[i, j]] -= lr * d_out[i] * inp[j];
            }
            self.b[i] -= lr * d_out[i];
        }
    }
}

fn near_centroid(rng: &mut rand_chacha::ChaCha8Rng) -> Array1<f32> {
    use rand::Rng;
    let noise = 0.06_f32;
    let raw: Vec<f32> = (0..DIM)
        .map(|_| (1.0 / DIM as f32) + (rng.random::<f32>() * 2.0 * noise - noise))
        .collect();
    let clamped: Vec<f32> = raw.iter().map(|&v| v.max(1e-3)).collect();
    let s: f32 = clamped.iter().sum();
    Array1::from_vec(clamped.iter().map(|&v| v / s).collect())
}

#[test]
fn fisher_rao_rfm_transports_mass_toward_the_mode_on_simplex() {
    let manifold = FisherRaoSimplex::default();

    // Single categorical mode, strictly interior so the log map is well posed.
    let mode = [0.80f32, 0.10, 0.10];
    let mut x1 = Array2::<f32>::zeros((1, DIM));
    for j in 0..DIM {
        x1[[0, j]] = mode[j];
    }

    let mut field = LinearTangentField::new();
    let cfg = SdFmTrainConfig {
        lr: 0.02,
        steps: 2_000,
        batch_size: 16,
        sample_steps: 0,
        seed: 7,
        ..Default::default()
    };

    train_riemannian_fm(&manifold, &mut field, &x1.view(), &cfg, near_centroid)
        .expect("training failed");

    // Integrate base samples forward and measure movement toward the mode.
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);
    let mode64 = Array1::from_vec(mode.iter().map(|&v| v as f64).collect());
    let dist_to_mode = |p: &Array1<f64>| -> f64 {
        let d = p - &mode64;
        d.dot(&d).sqrt()
    };

    let n = 16;
    let ode_steps = 40usize;
    let dt = 1.0 / ode_steps as f64;
    let mut base_sum = 0.0;
    let mut final_sum = 0.0;
    let mut on_simplex = 0;
    for _ in 0..n {
        let x0 = near_centroid(&mut rng).mapv(|v| v as f64);
        base_sum += dist_to_mode(&x0);
        let xf = integrate_fixed_manifold(
            OdeMethod::Heun,
            &manifold,
            &x0,
            0.0,
            dt,
            ode_steps,
            |x: &ArrayView1<f64>, t: f64| -> Array1<f64> {
                let v = field.eval(&x.mapv(|v| v as f32).view(), t as f32);
                v.mapv(|v| v as f64)
            },
        )
        .expect("integration failed");
        final_sum += dist_to_mode(&xf);
        let sum: f64 = xf.sum();
        if (sum - 1.0).abs() < 1e-6 && xf.iter().all(|&v| v >= -1e-9) {
            on_simplex += 1;
        }
    }

    // Every integrated point must stay on the probability simplex (the
    // Fisher-Rao exp map returns normalized squared coordinates).
    assert_eq!(on_simplex, n, "some integrated points left the simplex");
    // Training must move mass toward the mode: if FisherRaoSimplex were not
    // actually driving the loop, the field would have no consistent target.
    let base_avg = base_sum / n as f64;
    let final_avg = final_sum / n as f64;
    assert!(
        final_avg < base_avg * 0.75,
        "expected clear movement toward the mode: base={base_avg:.4}, final={final_avg:.4}"
    );
}
