use flowmatch::ode::OdeMethod;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_sd_fm_semidiscrete_linear_with_assignment, SdFmTrainAssignment, SdFmTrainConfig,
};
use ndarray::{Array1, Array2};
use wass::semidiscrete::SemidiscreteSgdConfig;

fn mean_sq_to_assigned_y(xs: &Array2<f32>, js: &[usize], y: &Array2<f32>) -> f32 {
    let n = xs.nrows();
    let d = xs.ncols();
    let mut s: f64 = 0.0;
    for i in 0..n {
        let j = js[i];
        for k in 0..d {
            let r = (xs[[i, k]] - y[[j, k]]) as f64;
            s += r * r;
        }
    }
    (s / (n as f64 * d as f64)) as f32
}

#[test]
fn sd_fm_training_improves_assignment_mse_vs_zero_field() {
    // Synthetic discrete support y_j (deterministic, no RNG).
    let n = 12usize;
    let d = 6usize;
    let mut y = Array2::<f32>::zeros((n, d));
    for j in 0..n {
        for k in 0..d {
            // Simple structured pattern; not “nice”, but deterministic.
            y[[j, k]] = (((j * 37 + k * 11) % 97) as f32 / 97.0) * 2.0 - 1.0;
        }
    }

    // Uniform weights.
    let b = Array1::<f32>::from_vec(vec![1.0; n]);

    let pot_cfg = SemidiscreteSgdConfig {
        epsilon: 0.0,
        lr: 0.8,
        steps: 1_200,
        batch_size: 512,
        seed: 7,
    };
    let fm_cfg = SdFmTrainConfig {
        lr: 1.2e-2,
        steps: 320,
        batch_size: 128,
        sample_steps: 30,
        seed: 9,
        t_schedule: TimestepSchedule::Uniform,
    };

    let trained = train_sd_fm_semidiscrete_linear_with_assignment(
        &y.view(),
        &b.view(),
        &pot_cfg,
        &fm_cfg,
        SdFmTrainAssignment::SemidiscretePotentials,
    )
    .expect("training should succeed");

    // Sample trajectories with the trained field.
    let n_samp = 128usize;
    let (x0s, xs_trained_euler, js) = trained
        .sample_with_x0_method(n_samp, 123, fm_cfg.sample_steps, OdeMethod::Euler)
        .expect("sampling should succeed");
    let (_x0s2, xs_trained_heun, js2) = trained
        .sample_with_x0_method(n_samp, 123, fm_cfg.sample_steps, OdeMethod::Heun)
        .expect("sampling should succeed");
    assert_eq!(js, js2, "same seed should pick same discrete indices");

    // Baseline: “zero field” leaves x(t)=x0, so final is far from y_j.
    let xs_zero = x0s;

    let mse_trained_euler = mean_sq_to_assigned_y(&xs_trained_euler, &js, &trained.y);
    let mse_trained_heun = mean_sq_to_assigned_y(&xs_trained_heun, &js, &trained.y);
    let mse_zero = mean_sq_to_assigned_y(&xs_zero, &js, &trained.y);

    // Two checks:
    // - trained is materially better than “do nothing”
    // - trained is absolutely not terrible in this tiny regime
    assert!(
        mse_trained_euler < 0.6 * mse_zero,
        "expected improvement (Euler): mse_trained={mse_trained_euler:.4} mse_zero={mse_zero:.4}"
    );
    assert!(
        mse_trained_heun < 0.7 * mse_zero,
        "expected improvement (Heun): mse_trained={mse_trained_heun:.4} mse_zero={mse_zero:.4}"
    );
    assert!(
        mse_trained_euler.is_finite() && mse_trained_heun.is_finite(),
        "mse should be finite"
    );
    assert!(
        mse_trained_euler < 2.0,
        "mse_trained too large for this toy regime: {mse_trained_euler:.4}"
    );
    assert!(
        mse_trained_heun < 2.0,
        "mse_trained too large for this toy regime: {mse_trained_heun:.4}"
    );
}

#[test]
fn sd_fm_categorical_assignment_baseline_is_not_worse_than_zero_field() {
    let n = 12usize;
    let d = 6usize;
    let mut y = Array2::<f32>::zeros((n, d));
    for j in 0..n {
        for k in 0..d {
            y[[j, k]] = (((j * 37 + k * 11) % 97) as f32 / 97.0) * 2.0 - 1.0;
        }
    }
    let b = Array1::<f32>::from_vec(vec![1.0; n]);

    // Potentials are not used for this assignment mode (kept for signature symmetry).
    let pot_cfg = SemidiscreteSgdConfig {
        epsilon: 0.0,
        lr: 0.1,
        steps: 1,
        batch_size: 1,
        seed: 0,
    };
    let fm_cfg = SdFmTrainConfig {
        lr: 1.2e-2,
        steps: 280,
        batch_size: 128,
        sample_steps: 30,
        seed: 9,
        t_schedule: TimestepSchedule::Uniform,
    };

    let trained = train_sd_fm_semidiscrete_linear_with_assignment(
        &y.view(),
        &b.view(),
        &pot_cfg,
        &fm_cfg,
        SdFmTrainAssignment::CategoricalFromB,
    )
    .expect("training should succeed");

    let n_samp = 128usize;
    let (x0s, xs_trained, js) = trained
        .sample_with_x0_method(n_samp, 123, fm_cfg.sample_steps, OdeMethod::Euler)
        .expect("sampling should succeed");

    let xs_zero = x0s;
    let mse_trained = mean_sq_to_assigned_y(&xs_trained, &js, &trained.y);
    let mse_zero = mean_sq_to_assigned_y(&xs_zero, &js, &trained.y);

    assert!(
        mse_trained < 0.95 * mse_zero,
        "expected at least some improvement vs zero field: mse_trained={mse_trained:.4} mse_zero={mse_zero:.4}"
    );
}
