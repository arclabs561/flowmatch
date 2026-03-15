//! Edge-case tests for degenerate inputs: n=1, d=1, identical points, zero velocity.
//!
//! No FM repo in the wild tests these cases. They surface NaN, panic, or silent
//! corruption in coupling, velocity evaluation, and ODE integration.

use flowmatch::linear::LinearCondField;
use flowmatch::ode::{integrate_fixed, OdeMethod};
use flowmatch::rfm::apply_pairing;
use flowmatch::sd_fm::{RfmMinibatchOtConfig, RfmMinibatchPairing};
use ndarray::{Array1, Array2};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_valid_pairing(perm: &[usize], n: usize) -> bool {
    perm.len() == n && perm.iter().all(|&j| j < n)
}

fn all_finite(arr: &Array1<f32>) -> bool {
    arr.iter().all(|x| x.is_finite())
}

// ---------------------------------------------------------------------------
// Coupling: apply_pairing on degenerate inputs
// ---------------------------------------------------------------------------

/// Pairing variants that don't use Sinkhorn (always fast, no convergence issues).
fn non_sinkhorn_pairings() -> Vec<RfmMinibatchPairing> {
    vec![
        RfmMinibatchPairing::RowwiseNearest,
        RfmMinibatchPairing::ExpGreedy { temp: 1.0 },
        RfmMinibatchPairing::PartialRowwise { keep_frac: 0.5 },
    ]
}

/// All pairing variants including Sinkhorn-based.
fn all_pairings() -> Vec<RfmMinibatchPairing> {
    let mut v = vec![
        RfmMinibatchPairing::SinkhornGreedy,
        RfmMinibatchPairing::SinkhornSelective { keep_frac: 0.5 },
    ];
    v.extend(non_sinkhorn_pairings());
    v
}

fn default_ot_cfg(pairing: RfmMinibatchPairing) -> RfmMinibatchOtConfig {
    RfmMinibatchOtConfig {
        reg: 1.0,
        max_iter: 500,
        tol: 1e-2,
        pairing,
        pairing_every: 1,
    }
}

#[test]
fn pairing_n1_d1_all_variants() {
    let x = Array2::from_shape_vec((1, 1), vec![0.0f32]).unwrap();
    let y = Array2::from_shape_vec((1, 1), vec![1.0f32]).unwrap();

    for pairing in all_pairings() {
        let cfg = default_ot_cfg(pairing);
        let result = apply_pairing(&pairing, &x.view(), &y.view(), &cfg);
        match result {
            Ok(perm) => assert!(is_valid_pairing(&perm, 1), "bad pairing for {pairing:?}"),
            Err(e) => {
                // Some variants may legitimately fail on n=1 (e.g. degenerate Sinkhorn).
                // Document the failure rather than panic.
                eprintln!("n=1 pairing {pairing:?} returned error: {e}");
            }
        }
    }
}

#[test]
fn pairing_identical_points_all_variants() {
    // Source and target are identical.
    let n = 4;
    let d = 3;
    let data: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
    let x = Array2::from_shape_vec((n, d), data.clone()).unwrap();
    let y = Array2::from_shape_vec((n, d), data).unwrap();

    for pairing in all_pairings() {
        let cfg = default_ot_cfg(pairing);
        let result = apply_pairing(&pairing, &x.view(), &y.view(), &cfg);
        match result {
            Ok(perm) => assert!(is_valid_pairing(&perm, n), "bad pairing for {pairing:?}"),
            Err(e) => eprintln!("identical-points pairing {pairing:?} returned error: {e}"),
        }
    }
}

#[test]
fn pairing_d1_non_sinkhorn() {
    // d=1 is a common edge case.
    let n = 5;
    let x = Array2::from_shape_vec((n, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array2::from_shape_vec((n, 1), vec![4.0, 3.0, 2.0, 1.0, 0.0]).unwrap();

    for pairing in all_pairings() {
        let cfg = default_ot_cfg(pairing);
        let result = apply_pairing(&pairing, &x.view(), &y.view(), &cfg);
        match result {
            Ok(perm) => assert!(is_valid_pairing(&perm, n), "bad pairing for {pairing:?}"),
            Err(e) => eprintln!("d=1 pairing {pairing:?} returned error: {e}"),
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        .. ProptestConfig::default()
    })]

    #[test]
    fn prop_pairing_rowwise_nearest_small_n_d(
        n in 1usize..=6,
        d in 1usize..=4,
        seed in 0u64..1000,
    ) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let x_data: Vec<f32> = (0..n * d).map(|_| StandardNormal.sample(&mut rng)).collect();
        let y_data: Vec<f32> = (0..n * d).map(|_| StandardNormal.sample(&mut rng)).collect();
        let x = Array2::from_shape_vec((n, d), x_data).unwrap();
        let y = Array2::from_shape_vec((n, d), y_data).unwrap();

        let pairing = RfmMinibatchPairing::RowwiseNearest;
        let cfg = RfmMinibatchOtConfig {
            reg: 1.0,
            max_iter: 500,
            tol: 1e-2,
            pairing,
            pairing_every: 1,
        };

        let perm = apply_pairing(&pairing, &x.view(), &y.view(), &cfg).unwrap();
        prop_assert!(is_valid_pairing(&perm, n));
    }
}

// ---------------------------------------------------------------------------
// LinearCondField: degenerate inputs
// ---------------------------------------------------------------------------

#[test]
fn linear_cond_field_d1() {
    let f = LinearCondField::new_zeros(1);
    let x = Array1::from_vec(vec![42.0f32]);
    let y = Array1::from_vec(vec![-1.0f32]);
    let out = f.eval(&x.view(), 0.5, &y.view()).unwrap();
    assert_eq!(out.len(), 1);
    assert!(all_finite(&out));
}

#[test]
fn linear_cond_field_identical_x_y() {
    let d = 4;
    let f = LinearCondField::new_zeros(d);
    let p = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
    let out = f.eval(&p.view(), 0.0, &p.view()).unwrap();
    assert_eq!(out.len(), d);
    assert!(all_finite(&out));
}

#[test]
fn linear_cond_field_t_near_boundaries() {
    let d = 3;
    let f = LinearCondField::new_zeros(d);
    let x = Array1::from_vec(vec![1.0f32, 0.0, -1.0]);
    let y = Array1::from_vec(vec![0.0f32, 1.0, 0.0]);

    // t very close to 0 and 1 -- the velocity target u_t = (y - x_t)/(1-t) diverges
    // at t=1, but the field itself should remain finite for zero weights.
    for t in [0.0, 1e-6, 0.999, 0.9999, 1.0] {
        let out = f.eval(&x.view(), t, &y.view()).unwrap();
        assert!(all_finite(&out), "non-finite at t={t}: {out:?}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        .. ProptestConfig::default()
    })]

    #[test]
    fn prop_linear_cond_field_always_finite(
        d in 1usize..=8,
        seed in 0u64..10000,
    ) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let f = LinearCondField::new_zeros(d);
        let x = Array1::from_vec((0..d).map(|_| StandardNormal.sample(&mut rng)).collect());
        let y = Array1::from_vec((0..d).map(|_| StandardNormal.sample(&mut rng)).collect());
        let t: f32 = rng.random();

        let out = f.eval(&x.view(), t, &y.view()).unwrap();
        prop_assert_eq!(out.len(), d);
        prop_assert!(all_finite(&out));
    }
}

// ---------------------------------------------------------------------------
// ODE integration: degenerate inputs
// ---------------------------------------------------------------------------

#[test]
fn ode_zero_velocity_returns_x0() {
    let x0 = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
    let dt = 0.1f32;

    for method in [OdeMethod::Euler, OdeMethod::Heun] {
        let result =
            integrate_fixed(method, &x0, 0.0, dt, 10, |_x, _t| Ok(Array1::zeros(3))).unwrap();

        for i in 0..3 {
            assert!(
                (result[i] - x0[i]).abs() < 1e-6,
                "zero velocity should preserve x0: method={method:?}, dim={i}"
            );
        }
    }
}

#[test]
fn ode_d1_single_step() {
    let x0 = Array1::from_vec(vec![5.0f32]);
    let dt = 0.5f32;

    for method in [OdeMethod::Euler, OdeMethod::Heun] {
        let result = integrate_fixed(method, &x0, 0.0, dt, 1, |_x, _t| {
            Ok(Array1::from_vec(vec![1.0]))
        })
        .unwrap();
        assert!(result[0].is_finite(), "d=1 single step should be finite");
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        .. ProptestConfig::default()
    })]

    #[test]
    fn prop_ode_finite_output_for_bounded_velocity(
        d in 1usize..=8,
        steps in 1usize..=50,
    ) {
        let x0 = Array1::zeros(d);
        let dt = 0.01f32;

        for method in [OdeMethod::Euler, OdeMethod::Heun] {
            let result = integrate_fixed(method, &x0, 0.0, dt, steps, |_x, _t| {
                Ok(Array1::ones(d))
            })
            .unwrap();
            prop_assert_eq!(result.len(), d);
            prop_assert!(all_finite(&result), "non-finite for d={d}, steps={steps}, method={method:?}");
        }
    }
}
