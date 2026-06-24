//! Fisher Flow Matching: Riemannian flow matching on the categorical simplex.
//!
//! Davis et al. (2024, NeurIPS), "Fisher Flow Matching for Generative Modeling
//! over Discrete Data", equips the probability simplex with the Fisher-Rao
//! metric and transports mass along its geodesics. Under the isometric
//! embedding `theta_i = sqrt(p_i)` the simplex becomes the positive orthant of
//! the unit sphere, where geodesics are great circles. This sidesteps the
//! discontinuous training targets that plague linear interpolation on the
//! simplex (Stark et al. 2024).
//!
//! There is no Fisher-Flow-specific code here: it is the existing generic
//! `train_riemannian_fm` (Chen & Lipman 2023) instantiated on a Fisher-Rao
//! manifold. The composition spans three crates and the f32/f64 seam is handled
//! entirely by the `skel::Manifold` trait boundary:
//! - `skel` -- the `Manifold` trait (exp/log/parallel transport/project)
//! - `infogeom` -- `FisherRaoSimplex` implementing `skel::Manifold` in f64
//! - `flowmatch` -- `train_riemannian_fm` + `integrate_fixed_manifold` in f32,
//!   casting to f64 only at the manifold calls.
//!
//! A two-layer MLP tangent field learns to route a near-uniform base
//! distribution to three categorical modes near the simplex vertices.

#[cfg(not(feature = "riemannian"))]
fn main() {
    eprintln!("This example requires `--features riemannian`.");
}

#[cfg(feature = "riemannian")]
fn main() {
    use flowmatch::ode::OdeMethod;
    use flowmatch::riemannian::{train_riemannian_fm, ManifoldVectorField};
    use flowmatch::riemannian_ode::integrate_fixed_manifold;
    use flowmatch::sd_fm::SdFmTrainConfig;
    use infogeom::FisherRaoSimplex;
    use ndarray::{Array1, Array2, ArrayView1};

    // -- Manifold: the Fisher-Rao simplex (categorical distributions) --
    let manifold = FisherRaoSimplex::default();

    // -- Target distributions: three categorical modes near the vertices of
    //    the 2-simplex (3 categories). Kept strictly interior so sqrt(p_i) and
    //    the log map stay well-conditioned. --
    const DIM: usize = 3;
    let targets: Vec<[f32; DIM]> = vec![[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]];
    let n_targets = targets.len();
    let mut x1_data = Array2::<f32>::zeros((n_targets, DIM));
    for (i, t) in targets.iter().enumerate() {
        for j in 0..DIM {
            x1_data[[i, j]] = t[j];
        }
    }

    // -- MLP tangent field (same generic field as the Poincare example) --
    const HIDDEN: usize = 32;

    /// Two-layer MLP operating in tangent space.
    /// Input: [x; t] (dim + 1) -> hidden (tanh) -> output (dim).
    struct MlpTangentField {
        dim: usize,
        hidden: usize,
        w1: Array2<f32>,
        b1: Array1<f32>,
        w2: Array2<f32>,
        b2: Array1<f32>,
    }

    impl MlpTangentField {
        fn new(dim: usize, hidden: usize, seed: u64) -> Self {
            let input_size = dim + 1;
            let mut state = seed;
            let scale1 = ((2.0 / (input_size + hidden) as f64).sqrt()) as f32;
            let scale2 = ((2.0 / (hidden + dim) as f64).sqrt()) as f32;
            let xorshift = |s: &mut u64| -> f32 {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
                ((*s as f32) / (u64::MAX as f32)) * 2.0 - 1.0
            };
            let w1 = Array2::from_shape_fn((hidden, input_size), |_| xorshift(&mut state) * scale1);
            let b1 = Array1::zeros(hidden);
            let w2 = Array2::from_shape_fn((dim, hidden), |_| xorshift(&mut state) * scale2);
            let b2 = Array1::zeros(dim);
            Self {
                dim,
                hidden,
                w1,
                b1,
                w2,
                b2,
            }
        }

        fn forward(&self, x: &ArrayView1<f32>, t: f32) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
            let input_size = self.dim + 1;
            let mut input = Array1::<f32>::zeros(input_size);
            for j in 0..self.dim {
                input[j] = x[j];
            }
            input[self.dim] = t;
            let pre = self.w1.dot(&input) + &self.b1;
            let h = pre.mapv(f32::tanh);
            let out = self.w2.dot(&h) + &self.b2;
            (pre, h, out)
        }
    }

    impl ManifoldVectorField for MlpTangentField {
        fn eval(&self, x: &ArrayView1<f32>, t: f32) -> Array1<f32> {
            let (_pre, _h, out) = self.forward(x, t);
            out
        }

        fn sgd_step(&mut self, x: &ArrayView1<f32>, t: f32, target: &ArrayView1<f32>, lr: f32) {
            let (_pre, h, out) = self.forward(x, t);
            let input_size = self.dim + 1;
            let mut input = Array1::<f32>::zeros(input_size);
            for j in 0..self.dim {
                input[j] = x[j];
            }
            input[self.dim] = t;
            let d_out = &out - target;
            let d_h = self.w2.t().dot(&d_out);
            for i in 0..self.dim {
                for j in 0..self.hidden {
                    self.w2[[i, j]] -= lr * d_out[i] * h[j];
                }
                self.b2[i] -= lr * d_out[i];
            }
            let d_pre = &d_h * &(1.0 - &h * &h);
            for i in 0..self.hidden {
                for j in 0..input_size {
                    self.w1[[i, j]] -= lr * d_pre[i] * input[j];
                }
                self.b1[i] -= lr * d_pre[i];
            }
        }
    }

    let mut field = MlpTangentField::new(DIM, HIDDEN, 42);

    let cfg = SdFmTrainConfig {
        lr: 0.01,
        steps: 4_000,
        batch_size: 16,
        sample_steps: 0,
        seed: 42,
        ..Default::default()
    };

    // -- Base distribution: near-uniform categorical (centroid of the simplex)
    //    with small noise, projected back onto the simplex. --
    let sample_x0 = |rng: &mut rand_chacha::ChaCha8Rng| -> Array1<f32> {
        use rand::Rng;
        let noise = 0.08_f32;
        let raw: Vec<f32> = (0..DIM)
            .map(|_| (1.0 / DIM as f32) + (rng.random::<f32>() * 2.0 * noise - noise))
            .collect();
        let clamped: Vec<f32> = raw.iter().map(|&v| v.max(1e-3)).collect();
        let s: f32 = clamped.iter().sum();
        Array1::from_vec(clamped.iter().map(|&v| v / s).collect())
    };

    train_riemannian_fm(&manifold, &mut field, &x1_data.view(), &cfg, sample_x0)
        .expect("training failed");

    println!(
        "Trained Fisher Flow Matching on the {DIM}-category simplex ({} steps, batch {}).",
        cfg.steps, cfg.batch_size
    );

    // -- Integrate base samples forward through the learned field --
    let n_samples = 12;
    let ode_steps = 50_usize;
    let dt = 1.0_f64 / (ode_steps as f64);

    let mut rng = {
        use rand::SeedableRng;
        rand_chacha::ChaCha8Rng::seed_from_u64(99)
    };
    let starts: Vec<Array1<f64>> = (0..n_samples)
        .map(|_| {
            use rand::Rng;
            let noise = 0.08_f64;
            let raw: Vec<f64> = (0..DIM)
                .map(|_| (1.0 / DIM as f64) + (rng.random::<f64>() * 2.0 * noise - noise))
                .collect();
            let clamped: Vec<f64> = raw.iter().map(|&v| v.max(1e-3)).collect();
            let s: f64 = clamped.iter().sum();
            Array1::from_vec(clamped.iter().map(|&v| v / s).collect())
        })
        .collect();

    let final_points: Vec<Array1<f64>> = starts
        .iter()
        .map(|x0| {
            integrate_fixed_manifold(
                OdeMethod::Heun,
                &manifold,
                x0,
                0.0,
                dt,
                ode_steps,
                |x: &ArrayView1<f64>, t: f64| -> Array1<f64> {
                    let x32 = x.mapv(|v| v as f32);
                    let v32 = field.eval(&x32.view(), t as f32);
                    v32.mapv(|v| v as f64)
                },
            )
            .expect("integration failed")
        })
        .collect();

    // -- Evaluation --
    let targets_f64: Vec<Array1<f64>> = targets
        .iter()
        .map(|t| Array1::from_vec(t.iter().map(|&v| v as f64).collect()))
        .collect();
    let nearest_mode_dist = |p: &Array1<f64>| -> f64 {
        targets_f64
            .iter()
            .map(|t| {
                let d = p - t;
                d.dot(&d).sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    };
    let is_simplex = |p: &Array1<f64>| -> bool {
        let sum: f64 = p.sum();
        (sum - 1.0).abs() < 1e-6 && p.iter().all(|&v| v >= -1e-9)
    };

    let base_avg: f64 = starts.iter().map(nearest_mode_dist).sum::<f64>() / n_samples as f64;
    let final_avg: f64 = final_points.iter().map(nearest_mode_dist).sum::<f64>() / n_samples as f64;
    let on_simplex = final_points.iter().filter(|p| is_simplex(p)).count();

    println!("  base   avg dist to nearest mode: {base_avg:.6}");
    println!("  final  avg dist to nearest mode: {final_avg:.6}");
    println!("  final points on simplex: {on_simplex}/{n_samples}");
    for (i, p) in final_points.iter().enumerate() {
        println!("  sample {i:2}: [{:.4}, {:.4}, {:.4}]", p[0], p[1], p[2]);
    }

    // The geodesic flow must keep every integrated point on the simplex...
    assert_eq!(
        on_simplex, n_samples,
        "some integrated points left the simplex ({on_simplex}/{n_samples})"
    );
    // ...and training must transport mass toward the modes (measurable
    // improvement over the base distribution, the contract the linear-FM e2e
    // tests also use). If the Fisher-Rao manifold did not actually drive the
    // integrator, the learned field would not move samples off the centroid.
    assert!(
        final_avg < base_avg,
        "training did not move samples toward the modes: base={base_avg:.6}, final={final_avg:.6}"
    );
}
