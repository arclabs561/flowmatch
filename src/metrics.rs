//! Evaluation utilities for flow models (small + explicit).
//!
//! These helpers are intentionally “glass box”:
//! - they do not hide solver settings
//! - they surface the exact scalar computed (and what it is *not*)

use crate::{Error, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Jensen–Shannon divergence between two nonnegative histograms (nats).
///
/// - Inputs may be unnormalized counts; this function normalizes them internally.
/// - Output is in **nats** (natural log), and satisfies \(0 \le JS \le \ln 2\).
///
/// This is a small wrapper around `logp::jensen_shannon_divergence` to keep flowmatch’s public
/// surface `f32`-first while reusing the ecosystem’s info-theory primitives.
pub fn jensen_shannon_divergence_histogram(p: &[f32], q: &[f32], tol: f32) -> Result<f32> {
    if p.is_empty() || q.is_empty() {
        return Err(Error::Domain("p and q must be non-empty"));
    }
    if p.len() != q.len() {
        return Err(Error::Shape("p and q must have the same length"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(Error::Domain("tol must be positive and finite"));
    }
    if p.iter().any(|&x| x < 0.0 || !x.is_finite()) || q.iter().any(|&x| x < 0.0 || !x.is_finite())
    {
        return Err(Error::Domain("p and q must be finite and nonnegative"));
    }

    let sp: f64 = p.iter().map(|&x| x as f64).sum();
    let sq: f64 = q.iter().map(|&x| x as f64).sum();
    if sp <= 0.0 || sq <= 0.0 {
        return Err(Error::Domain("p and q must have positive total mass"));
    }

    // Normalize + promote to f64 for logp.
    let mut pf64: Vec<f64> = Vec::with_capacity(p.len());
    let mut qf64: Vec<f64> = Vec::with_capacity(q.len());
    for i in 0..p.len() {
        pf64.push((p[i] as f64) / sp);
        qf64.push((q[i] as f64) / sq);
    }

    let js = logp::jensen_shannon_divergence(&pf64, &qf64, tol as f64)
        .map_err(|_| Error::Domain("logp::jensen_shannon_divergence failed"))?;
    Ok(js as f32)
}

/// Entropic OT cost between:
/// - an empirical sample set `xs` with **uniform** weights, and
/// - a discrete support `y` with weights `b` (will be normalized).
///
/// This returns the scalar transport cost \(\langle C, P\rangle\) for the entropic OT plan.
///
/// Notes:
/// - This is **not** Sinkhorn divergence (debiased). It’s just the entropic OT cost.
/// - This is useful as an evaluation signal for “are samples near the weighted support?”.
pub fn ot_cost_samples_to_weighted_support(
    xs: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    b: &ArrayView1<f32>,
    reg: f32,
    max_iter: usize,
    tol: f32,
) -> Result<f32> {
    let m = xs.nrows();
    let n = y.nrows();
    if m == 0 || n == 0 || xs.ncols() == 0 {
        return Err(Error::Domain("xs and y must be non-empty"));
    }
    if xs.ncols() != y.ncols() {
        return Err(Error::Shape("xs and y must have the same dimension"));
    }
    if b.len() != n {
        return Err(Error::Shape("b length must match y.nrows()"));
    }
    if b.iter().any(|&x| x < 0.0) {
        return Err(Error::Domain("b must be nonnegative"));
    }
    let bs = b.sum();
    if bs <= 0.0 {
        return Err(Error::Domain("b must have positive total mass"));
    }
    if !reg.is_finite() || reg <= 0.0 {
        return Err(Error::Domain("reg must be positive and finite"));
    }
    if max_iter == 0 {
        return Err(Error::Domain("max_iter must be >= 1"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(Error::Domain("tol must be positive and finite"));
    }

    let a = Array1::<f32>::from_elem(m, 1.0 / (m as f32));
    let b = b.to_owned() / bs;

    let cost: Array2<f32> = wass::euclidean_cost_matrix(&xs.to_owned(), &y.to_owned());
    let (_plan, dist, _iters) =
        wass::sinkhorn_log_with_convergence(&a, &b, &cost, reg, max_iter, tol)
            .map_err(|_| Error::Domain("sinkhorn did not converge"))?;
    Ok(dist)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn js_hist_is_symmetric_and_bounded() -> Result<()> {
        let p = vec![1.0f32, 2.0, 3.0, 4.0];
        let q = vec![4.0f32, 3.0, 2.0, 1.0];
        let js_pq = jensen_shannon_divergence_histogram(&p, &q, 1e-6)?;
        let js_qp = jensen_shannon_divergence_histogram(&q, &p, 1e-6)?;
        let ln2 = core::f32::consts::LN_2;
        assert!((js_pq - js_qp).abs() <= 1e-6);
        assert!(js_pq >= -1e-6);
        assert!(js_pq <= ln2 + 1e-4);
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 128,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_js_hist_basic_invariants(
            n in 1usize..128,
            seed in any::<u64>(),
            scale_p in 0.1f32..10.0f32,
            scale_q in 0.1f32..10.0f32,
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::{Distribution, StandardNormal};

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut p = vec![0.0f32; n];
            let mut q = vec![0.0f32; n];
            for i in 0..n {
                // nonnegative, allow zeros
                let a: f32 = StandardNormal.sample(&mut rng);
                let b: f32 = StandardNormal.sample(&mut rng);
                p[i] = (a.abs()) * scale_p;
                q[i] = (b.abs()) * scale_q;
            }
            // Ensure positive mass (rarely all zeros, but be explicit).
            prop_assume!(p.iter().any(|&x| x > 0.0));
            prop_assume!(q.iter().any(|&x| x > 0.0));

            let tol = 1e-6;
            let js_pq = jensen_shannon_divergence_histogram(&p, &q, tol).unwrap();
            let js_qp = jensen_shannon_divergence_histogram(&q, &p, tol).unwrap();
            let js_pp = jensen_shannon_divergence_histogram(&p, &p, tol).unwrap();

            let ln2 = core::f32::consts::LN_2;

            // Symmetry.
            prop_assert!((js_pq - js_qp).abs() <= 5e-6);
            // Bounds.
            prop_assert!(js_pq >= -1e-6);
            prop_assert!(js_pq <= ln2 + 1e-4);
            // Diagonal ~ 0.
            prop_assert!(js_pp <= 1e-5, "expected JS(p,p)=0; got {js_pp}");

            // Scaling invariance: JS(counts, counts) depends only on normalized distributions.
            let mut p2 = p.clone();
            for x in &mut p2 {
                *x *= 3.0;
            }
            let js_p2q = jensen_shannon_divergence_histogram(&p2, &q, tol).unwrap();
            prop_assert!((js_pq - js_p2q).abs() <= 5e-6);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_js_hist_error_contracts(
            n in 1usize..64,
        ) {
            let p = vec![1.0f32; n];
            let q = vec![1.0f32; n + 1];
            prop_assert!(jensen_shannon_divergence_histogram(&p, &q, 1e-6).is_err());
            prop_assert!(jensen_shannon_divergence_histogram(&p, &p, 0.0).is_err());
            prop_assert!(jensen_shannon_divergence_histogram(&p, &p, f32::NAN).is_err());

            let mut neg = vec![1.0f32; n];
            neg[0] = -1.0;
            prop_assert!(jensen_shannon_divergence_histogram(&neg, &p, 1e-6).is_err());

            let zeros = vec![0.0f32; n];
            prop_assert!(jensen_shannon_divergence_histogram(&zeros, &p, 1e-6).is_err());
        }
    }
}
