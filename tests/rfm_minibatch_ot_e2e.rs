use flowmatch::ode::OdeMethod;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use ndarray::{Array1, Array2};

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
fn rfm_minibatch_ot_trains_and_moves_toward_weighted_targets() {
    let n = 20usize;
    let d = 8usize;

    let mut y = Array2::<f32>::zeros((n, d));
    for j in 0..n {
        for k in 0..d {
            y[[j, k]] = (((j * 53 + k * 19) % 101) as f32 / 101.0) * 2.0 - 1.0;
        }
    }

    // Non-uniform weights.
    let mut b = Array1::<f32>::zeros(n);
    for j in 0..n {
        b[j] = 1.0 / ((j + 1) as f32).powf(0.9);
    }

    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 0.25,
        max_iter: 8000,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 4,
    };
    let fm_cfg = SdFmTrainConfig {
        lr: 1.0e-2,
        steps: 200,
        batch_size: 64,
        sample_steps: 25,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };

    let trained = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg).unwrap();

    // Sampling uses `CategoricalFromB` assignment (stored in model) + ODE integration.
    let (x0s, x1s, js) = trained
        .sample_with_x0_method(128, 999, fm_cfg.sample_steps, OdeMethod::Heun)
        .unwrap();

    let mse0 = mean_sq_to_assigned_y(&x0s, &js, &trained.y);
    let mse1 = mean_sq_to_assigned_y(&x1s, &js, &trained.y);

    assert!(
        mse1 < 0.8 * mse0,
        "expected movement toward assigned targets: mse1={mse1:.4} mse0={mse0:.4}"
    );
}
