use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{train_sd_fm_semidiscrete_linear, SdFmTrainConfig};
use ndarray::{Array1, Array2};
use wass::semidiscrete::SemidiscreteSgdConfig;

fn hist_from_assignments(js: &[usize], n: usize) -> Array1<f32> {
    let mut h = Array1::<f32>::zeros(n);
    for &j in js {
        h[j] += 1.0;
    }
    let s = h.sum();
    if s > 0.0 {
        h /= s;
    }
    h
}

fn l1(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

#[test]
fn sd_fm_matches_nonuniform_codebook_weights_and_moves_toward_targets() {
    // “Codebook”: discrete prototypes y_j, non-uniform weights b_j.
    let n = 24usize;
    let d = 8usize;
    let mut y = Array2::<f32>::zeros((n, d));
    for j in 0..n {
        for k in 0..d {
            y[[j, k]] = (((j * 53 + k * 19) % 101) as f32 / 101.0) * 2.0 - 1.0;
        }
    }

    // Heavy-tailed weights (Zipf-ish), then normalize.
    let mut b = Array1::<f32>::zeros(n);
    for j in 0..n {
        b[j] = 1.0 / ((j + 1) as f32).powf(0.9);
    }

    let pot_cfg = SemidiscreteSgdConfig {
        epsilon: 0.0,
        lr: 0.8,
        steps: 1_000,
        batch_size: 512,
        seed: 17,
    };
    let fm_cfg = SdFmTrainConfig {
        lr: 1.0e-2,
        steps: 260,
        batch_size: 128,
        sample_steps: 30,
        seed: 19,
        t_schedule: TimestepSchedule::Uniform,
    };

    let trained = train_sd_fm_semidiscrete_linear(&y.view(), &b.view(), &pot_cfg, &fm_cfg)
        .expect("training should succeed");

    let n_samp = 256usize;
    let (x0s, x1s, js) = trained
        .sample_with_x0(n_samp, 999, fm_cfg.sample_steps)
        .expect("sampling should succeed");

    let hist = hist_from_assignments(&js, n);
    let l1_hist = l1(&hist, &trained.b);

    // Distance-to-assigned-target should improve from x0 to x1 (on average).
    let mut mse0: f64 = 0.0;
    let mut mse1: f64 = 0.0;
    for i in 0..n_samp {
        let j = js[i];
        for k in 0..d {
            let r0 = (x0s[[i, k]] - trained.y[[j, k]]) as f64;
            let r1 = (x1s[[i, k]] - trained.y[[j, k]]) as f64;
            mse0 += r0 * r0;
            mse1 += r1 * r1;
        }
    }
    mse0 /= n_samp as f64 * d as f64;
    mse1 /= n_samp as f64 * d as f64;

    // Two checks:
    // - assignment frequencies roughly match the requested weights
    // - the flow moved samples toward their assigned prototype
    assert!(
        l1_hist < 0.35,
        "assignment histogram too far from target: L1={l1_hist:.3}"
    );
    assert!(
        mse1 < 0.7 * mse0,
        "expected movement toward targets: mse1={mse1:.4} mse0={mse0:.4}"
    );
}
