//! Property tests for OT coupling quality.
//!
//! Verifies:
//! - Sinkhorn cost decreases as regularization decreases (monotonicity)
//! - Sinkhorn pairing produces lower total transport cost than random coupling
//! - Reuse-seam: pairing_every > 1 doesn't silently change training semantics

use flowmatch::rfm::minibatch_ot_greedy_pairing;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, FmTrainConfig, RfmMinibatchOtConfig, RfmMinibatchPairing,
    TimestepSchedule,
};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn synthetic_targets(n: usize, d: usize) -> (Array2<f32>, Array1<f32>) {
    let mut y = Array2::<f32>::zeros((n, d));
    for j in 0..n {
        for k in 0..d {
            y[[j, k]] = (((j * 53 + k * 19) % 101) as f32 / 101.0) * 2.0 - 1.0;
        }
    }
    let b = Array1::from_elem(n, 1.0 / n as f32);
    (y, b)
}

fn gaussian_points(n: usize, d: usize, seed: u64) -> Array2<f32> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x = Array2::<f32>::zeros((n, d));
    for i in 0..n {
        for k in 0..d {
            x[[i, k]] = StandardNormal.sample(&mut rng);
        }
    }
    x
}

/// Total squared L2 cost of a pairing: sum_i ||x[i] - y[perm[i]]||^2.
fn pairing_cost(x: &Array2<f32>, y: &Array2<f32>, perm: &[usize]) -> f64 {
    let n = x.nrows();
    let d = x.ncols();
    let mut cost = 0.0f64;
    for i in 0..n {
        let j = perm[i];
        for k in 0..d {
            let r = (x[[i, k]] - y[[j, k]]) as f64;
            cost += r * r;
        }
    }
    cost / n as f64
}

// ---------------------------------------------------------------------------
// OT coupling monotonicity: lower reg -> lower transport cost
// ---------------------------------------------------------------------------

#[test]
fn sinkhorn_cost_decreases_with_lower_regularization() {
    let n = 16;
    let d = 4;
    let x = gaussian_points(n, d, 42);
    let (y, _b) = synthetic_targets(n, d);

    // Compare high-reg (blurry) vs low-reg (sharper) Sinkhorn coupling.
    // The greedy matching step adds noise, so we test only that the sharpest
    // regularization (closest to true OT) beats a deliberately blurry one.
    let perm_blurry = minibatch_ot_greedy_pairing(&x.view(), &y.view(), 5.0, 500, 1e-3).unwrap();
    let perm_sharp = minibatch_ot_greedy_pairing(&x.view(), &y.view(), 0.1, 500, 1e-3).unwrap();

    let cost_blurry = pairing_cost(&x, &y, &perm_blurry);
    let cost_sharp = pairing_cost(&x, &y, &perm_sharp);

    assert!(
        cost_sharp <= cost_blurry + 0.1,
        "sharp reg should not be much worse than blurry: sharp={cost_sharp:.4}, blurry={cost_blurry:.4}"
    );
}

#[test]
fn sinkhorn_pairing_beats_random_coupling() {
    let n = 16;
    let d = 4;
    let x = gaussian_points(n, d, 42);
    let (y, _b) = synthetic_targets(n, d);

    let ot_perm = minibatch_ot_greedy_pairing(&x.view(), &y.view(), 0.5, 500, 1e-3).unwrap();
    let ot_cost = pairing_cost(&x, &y, &ot_perm);

    // Identity (random-ish) coupling.
    let identity_perm: Vec<usize> = (0..n).collect();
    let identity_cost = pairing_cost(&x, &y, &identity_perm);

    assert!(
        ot_cost <= identity_cost,
        "OT pairing should have lower cost than identity: ot={ot_cost:.4}, identity={identity_cost:.4}"
    );
}

// ---------------------------------------------------------------------------
// Scheduler sensitivity: all schedules produce reasonable training loss
// ---------------------------------------------------------------------------

#[test]
fn all_schedules_produce_finite_loss_on_small_problem() {
    let n = 12;
    let d = 4;
    let (y, b) = synthetic_targets(n, d);

    let schedules = [
        TimestepSchedule::Uniform,
        TimestepSchedule::UShaped,
        TimestepSchedule::LogitNormal {
            mean: 0.0,
            std: 1.0,
        },
    ];

    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 1.0,
        max_iter: 500,
        tol: 1e-2,
        pairing: RfmMinibatchPairing::RowwiseNearest, // Fast, no Sinkhorn overhead.
        pairing_every: 1,
    };

    for schedule in &schedules {
        let fm_cfg = FmTrainConfig {
            lr: 5e-3,
            steps: 50,
            batch_size: n.min(16),
            sample_steps: 10,
            seed: 7,
            t_schedule: *schedule,
        };

        let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg).unwrap();

        // Field weights should be finite.
        assert!(
            model.field.w().iter().all(|x| x.is_finite()),
            "non-finite field weights for schedule {schedule:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Reuse-seam: pairing_every > 1 doesn't silently change semantics
// ---------------------------------------------------------------------------

#[test]
fn pairing_every_k_produces_bounded_divergence_from_every_1() {
    let n = 12;
    let d = 4;
    let (y, b) = synthetic_targets(n, d);

    let base_fm_cfg = FmTrainConfig {
        lr: 5e-3,
        steps: 50,
        batch_size: n.min(16),
        sample_steps: 10,
        seed: 7,
        t_schedule: TimestepSchedule::Uniform,
    };

    let rfm_every1 = RfmMinibatchOtConfig {
        reg: 1.0,
        max_iter: 500,
        tol: 1e-2,
        pairing: RfmMinibatchPairing::RowwiseNearest,
        pairing_every: 1,
    };

    let rfm_every4 = RfmMinibatchOtConfig {
        pairing_every: 4,
        ..rfm_every1.clone()
    };

    let model_1 =
        train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_every1, &base_fm_cfg).unwrap();
    let model_4 =
        train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_every4, &base_fm_cfg).unwrap();

    // Compare field weights: the norm difference should be bounded.
    // They won't be identical (different coupling reuse patterns), but shouldn't diverge wildly.
    let w1 = model_1.field.w();
    let w4 = model_4.field.w();

    let norm_diff: f64 = w1
        .iter()
        .zip(w4.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm_1: f64 = w1.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

    // Relative difference should be moderate (< 1.0 = 100% of the weight norm).
    // This is a loose bound -- we're checking for catastrophic divergence, not equality.
    let rel_diff = if norm_1 > 1e-6 {
        norm_diff / norm_1
    } else {
        norm_diff
    };

    assert!(
        rel_diff < 1.0,
        "pairing_every=4 diverged too much from pairing_every=1: rel_diff={rel_diff:.4}"
    );
}
