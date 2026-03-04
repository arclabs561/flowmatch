//! Class-conditional flow matching on 2D synthetic data.
//!
//! Trains one linear conditional field per class, then generates samples
//! conditioned on each class label and checks they land near the correct
//! cluster center.
//!
//! - Class 0: cluster at (-3, 0)
//! - Class 1: cluster at ( 3, 0)
//! - Class 2: cluster at ( 0, 3)
//!
//! Source distribution: standard normal for all classes.
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_conditional_2d
//! ```

use flowmatch::linear::LinearCondField;
use flowmatch::ode::{integrate_fixed, OdeMethod};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

const CENTERS: [[f32; 2]; 3] = [[-3.0, 0.0], [3.0, 0.0], [0.0, 3.0]];
const N_PER_CLASS: usize = 200;
const NOISE: f32 = 0.3;
const TRAIN_STEPS: usize = 6_000;
const LR: f32 = 3e-3;
const SAMPLE_STEPS: usize = 30;
const N_EVAL: usize = 200;

/// Generate target samples for one class: Gaussian cluster around `center`.
fn sample_cluster(center: [f32; 2], n: usize, rng: &mut ChaCha8Rng) -> Array2<f32> {
    let noise_dist = Normal::new(0.0f32, NOISE).unwrap();
    let mut out = Array2::<f32>::zeros((n, 2));
    for i in 0..n {
        out[[i, 0]] = center[0] + noise_dist.sample(rng);
        out[[i, 1]] = center[1] + noise_dist.sample(rng);
    }
    out
}

/// Train a `LinearCondField` on (source, target) pairs for one class.
///
/// Uses the standard conditional FM objective: sample t ~ U(0,1),
/// form x_t = (1-t)*x0 + t*y, regress v_theta(x_t, t; y) toward u = y - x0.
fn train_class_field(target: &Array2<f32>, seed: u64) -> LinearCondField {
    let n = target.nrows();
    let mut field = LinearCondField::new_zeros(2);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for _ in 0..TRAIN_STEPS {
        // Sample a random target point.
        let idx = rand::Rng::random_range(&mut rng, 0..n);
        let y = target.row(idx);

        // Sample x0 ~ N(0, I).
        let x0_0: f32 = StandardNormal.sample(&mut rng);
        let x0_1: f32 = StandardNormal.sample(&mut rng);
        let x0 = Array1::from_vec(vec![x0_0, x0_1]);

        // Sample t ~ U(0,1), clamped away from boundaries.
        let t: f32 = rand::Rng::random::<f32>(&mut rng).clamp(1e-5, 1.0 - 1e-5);

        // Interpolant: x_t = (1-t)*x0 + t*y.
        let xt = Array1::from_vec(vec![
            (1.0 - t) * x0[0] + t * y[0],
            (1.0 - t) * x0[1] + t * y[1],
        ]);

        // Target velocity: u = y - x0.
        let u = Array1::from_vec(vec![y[0] - x0[0], y[1] - x0[1]]);

        field.sgd_step(&xt.view(), t, &y, &u.view(), LR);
    }
    field
}

/// Generate samples from a trained field by integrating from N(0,I).
///
/// Uses the target cluster mean as the conditioning signal y (since at
/// inference we condition on the class, not on individual target points).
fn generate_samples(field: &LinearCondField, center: [f32; 2], n: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dt = 1.0f32 / (SAMPLE_STEPS as f32);
    let y = Array1::from_vec(vec![center[0], center[1]]);

    let mut out = Array2::<f32>::zeros((n, 2));
    for i in 0..n {
        let x0 = Array1::from_vec(vec![
            StandardNormal.sample(&mut rng),
            StandardNormal.sample(&mut rng),
        ]);
        let x1 = integrate_fixed(OdeMethod::Euler, &x0, 0.0, dt, SAMPLE_STEPS, |xt, t| {
            field.eval(xt, t, &y.view())
        });
        out[[i, 0]] = x1[0];
        out[[i, 1]] = x1[1];
    }
    out
}

fn dist(a: &[f32], b: &[f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn main() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Generate target data per class.
    let targets: Vec<Array2<f32>> = CENTERS
        .iter()
        .map(|c| sample_cluster(*c, N_PER_CLASS, &mut rng))
        .collect();

    // Train one field per class.
    let fields: Vec<LinearCondField> = targets
        .iter()
        .enumerate()
        .map(|(c, t)| {
            let f = train_class_field(t, 100 + c as u64);
            println!("Trained class {c} field.");
            f
        })
        .collect();

    // Generate and evaluate.
    let mut total_correct = 0usize;
    let mut total = 0usize;

    println!("\nPer-class evaluation ({N_EVAL} samples each):");
    for (c, field) in fields.iter().enumerate() {
        let samples = generate_samples(field, CENTERS[c], N_EVAL, 500 + c as u64);

        // Mean distance to true center.
        let mean_dist: f32 = (0..N_EVAL)
            .map(|i| dist(&[samples[[i, 0]], samples[[i, 1]]], &CENTERS[c]))
            .sum::<f32>()
            / N_EVAL as f32;

        // Accuracy: fraction closer to correct center than any other.
        let correct = (0..N_EVAL)
            .filter(|&i| {
                let s = [samples[[i, 0]], samples[[i, 1]]];
                let d_own = dist(&s, &CENTERS[c]);
                CENTERS
                    .iter()
                    .enumerate()
                    .all(|(k, ck)| k == c || dist(&s, ck) > d_own)
            })
            .count();

        total_correct += correct;
        total += N_EVAL;

        println!(
            "  class {c} center={:?}: mean_dist={mean_dist:.3}, accuracy={}/{N_EVAL} ({:.1}%)",
            CENTERS[c],
            correct,
            100.0 * correct as f32 / N_EVAL as f32,
        );
    }

    let overall = 100.0 * total_correct as f32 / total as f32;
    println!("\nOverall accuracy: {total_correct}/{total} ({overall:.1}%)");
}
