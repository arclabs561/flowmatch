//! OT-CFM training loop using the low-level coupling + training step API.
//!
//! Trains a linear conditional vector field on a 2D four-cluster target,
//! using OT-coupled pairings for straighter flow trajectories.

use flowmatch::linear::LinearCondField;
use flowmatch::ode::{integrate_fixed, OdeMethod};
use flowmatch::ot_cfm::{ot_cfm_training_step, OtCfmConfig};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 32;
    let dim = 2;
    let steps = 200;
    let lr = 0.01_f32;

    // Target: 4 cluster centers in 2D.
    let centers = [[2.0_f32, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]];

    let ot_cfg = OtCfmConfig {
        reg: 1.0,
        max_sinkhorn_iter: 100,
        sinkhorn_tol: 1e-2,
    };

    let mut field = LinearCondField::new_zeros(dim);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    println!("OT-CFM training: {dim}D, batch={batch_size}, steps={steps}");
    println!();

    for step in 0..steps {
        // Source: noise x0 ~ N(0, I).
        let mut x0 = Array2::<f32>::zeros((batch_size, dim));
        for i in 0..batch_size {
            for j in 0..dim {
                x0[[i, j]] = StandardNormal.sample(&mut rng);
            }
        }

        // Target: cluster samples.
        let mut x1 = Array2::<f32>::zeros((batch_size, dim));
        for i in 0..batch_size {
            let c = &centers[i % 4];
            for j in 0..dim {
                x1[[i, j]] = c[j]
                    + 0.3 * {
                        let v: f64 = StandardNormal.sample(&mut rng);
                        v as f32
                    };
            }
        }

        // Timesteps t ~ U[0,1].
        let t: Vec<f32> = (0..batch_size)
            .map(|_| rng.random_range(0.0_f32..1.0))
            .collect();

        // OT-CFM training step: coupling + interpolation + velocity targets.
        let targets = ot_cfm_training_step(&x0.view(), &x1.view(), &t, &ot_cfg)?;

        // Update field with SGD on each sample.
        for i in 0..batch_size {
            let xt_row = targets.x_t.row(i);
            let j = targets.coupling[i];
            let y_row = x1.row(j);
            let u_row = targets.u_t.row(i);
            field.sgd_step(&xt_row, t[i], &y_row, &u_row, lr / batch_size as f32)?;
        }

        if step % 50 == 0 || step == steps - 1 {
            let avg_dist = evaluate_sampling(&field, &centers, &mut rng)?;
            println!("Step {step:>3}: avg dist to center = {avg_dist:.4}");
        }
    }

    Ok(())
}

fn evaluate_sampling(
    field: &LinearCondField,
    centers: &[[f32; 2]; 4],
    rng: &mut ChaCha8Rng,
) -> Result<f32, Box<dyn std::error::Error>> {
    let n_eval = 32;
    let ode_steps = 20;
    let dt = 1.0 / ode_steps as f32;
    let dim = 2;
    let mut total_dist = 0.0_f32;

    for i in 0..n_eval {
        let mut x = Array1::<f32>::zeros(dim);
        for j in 0..dim {
            x[j] = StandardNormal.sample(rng);
        }

        let c_idx = i % 4;
        let y = Array1::from_vec(centers[c_idx].to_vec());

        let result = integrate_fixed(OdeMethod::Heun, &x, 0.0, dt, ode_steps, |x_arr, t_val| {
            field.eval(x_arr, t_val, &y.view())
        })?;

        let dist: f32 = result
            .iter()
            .zip(centers[c_idx].iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        total_dist += dist;
    }

    Ok(total_dist / n_eval as f32)
}
