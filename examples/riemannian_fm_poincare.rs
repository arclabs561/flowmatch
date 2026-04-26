//! Riemannian Flow Matching on the Poincare ball with an MLP vector field.
//!
//! Trains a two-layer MLP tangent-space vector field to transport a base distribution
//! (small ball near the origin) to a multi-modal set of target points on the Poincare
//! disk (2D, curvature c=1).
//!
//! A linear field can only produce a single "average" output direction for each (x, t),
//! which limits it to collapsing all samples toward the centroid of the targets. An MLP
//! has enough capacity to partition the input space and route different starting points
//! toward different modes, making it suitable for multi-modal target distributions on
//! manifolds.
//!
//! Architecture: input [x; t] (dim+1) -> hidden layer (tanh) -> output (dim).
//! All operations happen in the tangent space at x; the Riemannian structure
//! (exp map, parallel transport) is handled by the training loop externally.
//!
//! Connects three crates:
//! - `skel` -- the `Manifold` trait (exp/log/parallel transport/project)
//! - `hyperball` -- `PoincareBall<f64>` implementing `skel::Manifold`
//! - `flowmatch` -- `train_riemannian_fm` + `integrate_fixed_manifold`

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
    use hyperball::PoincareBall;
    use ndarray::{Array1, Array2, ArrayView1};

    // -- Manifold --
    let manifold = PoincareBall::<f64>::new(1.0);

    // -- Target points (well inside the unit ball) --
    // Six targets spread around the disk to exercise multi-modal capacity.
    let targets: Vec<[f32; 2]> = vec![
        [0.4, 0.3],
        [-0.3, 0.5],
        [-0.4, -0.2],
        [0.2, -0.5],
        [0.5, -0.1],
        [-0.1, 0.6],
    ];
    let n_targets = targets.len();
    let mut x1_data = Array2::<f32>::zeros((n_targets, 2));
    for (i, t) in targets.iter().enumerate() {
        x1_data[[i, 0]] = t[0];
        x1_data[[i, 1]] = t[1];
    }

    // -- MLP tangent field --
    const DIM: usize = 2;
    const HIDDEN: usize = 32;

    /// Two-layer MLP operating in tangent space.
    /// Input: [x; t] (dim + 1) -> hidden (tanh) -> output (dim).
    struct MlpTangentField {
        dim: usize,
        hidden: usize,
        // Layer 1: (dim+1) -> hidden
        w1: Array2<f32>, // hidden x (dim+1), row-major
        b1: Array1<f32>, // hidden
        // Layer 2: hidden -> dim
        w2: Array2<f32>, // dim x hidden, row-major
        b2: Array1<f32>, // dim
    }

    impl MlpTangentField {
        fn new(dim: usize, hidden: usize, seed: u64) -> Self {
            let input_size = dim + 1;
            let mut state = seed;

            // Xavier-scale initialization: std = sqrt(2 / (fan_in + fan_out))
            let scale1 = ((2.0 / (input_size + hidden) as f64).sqrt()) as f32;
            let scale2 = ((2.0 / (hidden + dim) as f64).sqrt()) as f32;

            let xorshift = |s: &mut u64| -> f32 {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
                // Map to [-1, 1]
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

        /// Forward pass returning (pre_activation, post_activation, output).
        fn forward(&self, x: &ArrayView1<f32>, t: f32) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
            // Build input vector [x; t]
            let input_size = self.dim + 1;
            let mut input = Array1::<f32>::zeros(input_size);
            for j in 0..self.dim {
                input[j] = x[j];
            }
            input[self.dim] = t;

            // Layer 1: pre = W1 * input + b1
            let pre = self.w1.dot(&input) + &self.b1;
            // tanh activation
            let h = pre.mapv(f32::tanh);

            // Layer 2: out = W2 * h + b2
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

            // Build input vector [x; t]
            let input_size = self.dim + 1;
            let mut input = Array1::<f32>::zeros(input_size);
            for j in 0..self.dim {
                input[j] = x[j];
            }
            input[self.dim] = t;

            // dL/dout = pred - target (gradient of 0.5 * ||pred - target||^2)
            let d_out = &out - target;

            // -- Backprop through layer 2 (before updating w2) --
            // dL/dh = W2^T * d_out  (hidden)
            let d_h = self.w2.t().dot(&d_out);

            // -- Layer 2 gradients --
            // dL/dW2 = d_out (dim) outer h (hidden) -> (dim x hidden)
            // dL/db2 = d_out
            for i in 0..self.dim {
                for j in 0..self.hidden {
                    self.w2[[i, j]] -= lr * d_out[i] * h[j];
                }
                self.b2[i] -= lr * d_out[i];
            }

            // -- tanh derivative: dL/dpre = dL/dh * (1 - tanh^2(pre)) = dL/dh * (1 - h^2) --
            let d_pre = &d_h * &(1.0 - &h * &h);

            // -- Layer 1 gradients --
            // dL/dW1 = d_pre (hidden) outer input (input_size) -> (hidden x input_size)
            // dL/db1 = d_pre
            for i in 0..self.hidden {
                for j in 0..input_size {
                    self.w1[[i, j]] -= lr * d_pre[i] * input[j];
                }
                self.b1[i] -= lr * d_pre[i];
            }
        }
    }

    let mut field = MlpTangentField::new(DIM, HIDDEN, 42);

    // -- Training config --
    // More steps than the linear case: the MLP has ~(3*32 + 32*2 + 32 + 2) = 194 parameters.
    let cfg = SdFmTrainConfig {
        lr: 0.01,
        steps: 4_000,
        batch_size: 16,
        sample_steps: 0, // we sample manually below
        seed: 42,
        ..Default::default()
    };

    // -- Base distribution sampler: small Gaussian near origin, projected into ball --
    let sample_x0 = |rng: &mut rand_chacha::ChaCha8Rng| -> Array1<f32> {
        use rand::Rng;
        let r = 0.05_f32;
        Array1::from_vec(vec![
            rng.random::<f32>() * 2.0 * r - r,
            rng.random::<f32>() * 2.0 * r - r,
        ])
    };

    // -- Train --
    train_riemannian_fm(&manifold, &mut field, &x1_data.view(), &cfg, sample_x0)
        .expect("training failed");

    println!(
        "Training complete ({} steps, batch_size {}, hidden_size {})",
        cfg.steps, cfg.batch_size, HIDDEN
    );

    // -- Sample with both Euler and Heun --
    let n_samples = 12;
    let ode_steps = 50_usize;
    let dt = 1.0_f64 / (ode_steps as f64);

    let mut rng = {
        use rand::SeedableRng;
        rand_chacha::ChaCha8Rng::seed_from_u64(99)
    };

    // Generate starting points (same for both methods).
    let starts: Vec<Array1<f64>> = (0..n_samples)
        .map(|_| {
            use rand::Rng;
            let r = 0.05_f64;
            Array1::from_vec(vec![
                rng.random::<f64>() * 2.0 * r - r,
                rng.random::<f64>() * 2.0 * r - r,
            ])
        })
        .collect();

    // Velocity wrapper: field operates in f32, ODE integrator in f64.
    let mut euler_results = Vec::new();
    let mut heun_results = Vec::new();

    for x0 in &starts {
        let e = integrate_fixed_manifold(
            OdeMethod::Euler,
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
        .expect("Euler integration failed");
        euler_results.push(e);

        let h = integrate_fixed_manifold(
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
        .expect("Heun integration failed");
        heun_results.push(h);
    }

    // -- Evaluation --
    let targets_f64: Vec<Array1<f64>> = targets
        .iter()
        .map(|t| Array1::from_vec(vec![t[0] as f64, t[1] as f64]))
        .collect();

    let nearest_target_dist = |point: &Array1<f64>| -> f64 {
        targets_f64
            .iter()
            .map(|t| {
                let d = point - t;
                d.dot(&d).sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    };

    let in_ball = |point: &Array1<f64>| -> bool { point.dot(point) < 1.0 };

    // (a) Check all final points stay inside the ball.
    let euler_in = euler_results.iter().filter(|p| in_ball(p)).count();
    let heun_in = heun_results.iter().filter(|p| in_ball(p)).count();

    // (b) Average distance to nearest target.
    let euler_avg: f64 = euler_results
        .iter()
        .map(nearest_target_dist)
        .sum::<f64>()
        / n_samples as f64;
    let heun_avg: f64 = heun_results
        .iter()
        .map(nearest_target_dist)
        .sum::<f64>()
        / n_samples as f64;

    println!();
    println!("--- Euler ({ode_steps} steps) ---");
    println!("  points inside ball: {euler_in}/{n_samples}");
    println!("  avg dist to nearest target: {euler_avg:.6}");
    for (i, p) in euler_results.iter().enumerate() {
        println!("  sample {i:2}: [{:.4}, {:.4}]", p[0], p[1]);
    }

    println!();
    println!("--- Heun ({ode_steps} steps) ---");
    println!("  points inside ball: {heun_in}/{n_samples}");
    println!("  avg dist to nearest target: {heun_avg:.6}");
    for (i, p) in heun_results.iter().enumerate() {
        println!("  sample {i:2}: [{:.4}, {:.4}]", p[0], p[1]);
    }

    println!();
    if heun_avg < euler_avg {
        println!(
            "Heun is closer to targets by {:.6} (avg Euclidean dist).",
            euler_avg - heun_avg
        );
    } else {
        println!(
            "Euler is closer to targets by {:.6} (avg Euclidean dist).",
            heun_avg - euler_avg
        );
    }

    // Sanity: all samples must remain inside the Poincare ball.
    assert!(
        euler_in == n_samples,
        "some Euler samples escaped the ball ({euler_in}/{n_samples})"
    );
    assert!(
        heun_in == n_samples,
        "some Heun samples escaped the ball ({heun_in}/{n_samples})"
    );
}
