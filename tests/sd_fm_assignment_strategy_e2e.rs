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
fn sd_fm_training_with_categorical_assignment_moves_toward_targets() {
    let n = 16usize;
    let d = 8usize;
    let mut y = Array2::<f32>::zeros((n, d));
    for j in 0..n {
        for k in 0..d {
            y[[j, k]] = (((j * 37 + k * 11) % 97) as f32 / 97.0) * 2.0 - 1.0;
        }
    }

    // Non-uniform b.
    let mut b = Array1::<f32>::zeros(n);
    for j in 0..n {
        b[j] = 1.0 / ((j + 1) as f32).powf(0.7);
    }

    let pot_cfg = SemidiscreteSgdConfig {
        epsilon: 0.0,
        lr: 0.8,
        steps: 400,
        batch_size: 256,
        seed: 7,
    };
    let fm_cfg = SdFmTrainConfig {
        lr: 1.0e-2,
        steps: 200,
        batch_size: 128,
        sample_steps: 25,
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

    let (x0s, x1s, js) = trained
        .sample_with_x0_method(128, 123, fm_cfg.sample_steps, OdeMethod::Heun)
        .expect("sampling should succeed");

    let mse0 = mean_sq_to_assigned_y(&x0s, &js, &trained.y);
    let mse1 = mean_sq_to_assigned_y(&x1s, &js, &trained.y);
    assert!(
        mse1 < 0.8 * mse0,
        "expected movement toward targets: mse1={mse1:.4} mse0={mse0:.4}"
    );
}
