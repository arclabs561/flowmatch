//! Simplex-based helpers (for “discrete FM on the simplex” style methods).
//!
//! This module is intentionally minimal: it provides **explicit** utilities for
//! (1) validating simplex constraints and (2) sampling basic relaxations.
//!
//! Public invariant: we do **not** silently normalize in methods named like “validate”.
//! If we normalize, the function name says so (`normalize_*`).

use crate::{Error, Result};

/// Check whether `p` lies on the probability simplex (within `tol`).
pub fn validate_simplex(p: &[f32], tol: f32) -> Result<()> {
    if p.is_empty() {
        return Err(Error::Domain("simplex vector must be non-empty"));
    }
    if !(tol >= 0.0) || !tol.is_finite() {
        return Err(Error::Domain("tol must be finite and >= 0"));
    }
    if p.iter().any(|&x| !x.is_finite()) {
        return Err(Error::Domain("simplex vector contains non-finite values"));
    }
    if p.iter().any(|&x| x < -tol) {
        return Err(Error::Domain("simplex vector has negative entries"));
    }
    let s: f32 = p.iter().sum();
    if (s - 1.0).abs() > tol {
        return Err(Error::Domain(
            "simplex vector does not sum to 1 (within tol)",
        ));
    }
    Ok(())
}

/// Explicit normalization to the simplex via `p_i / sum(p)`, with checks.
pub fn normalize_simplex(p: &[f32]) -> Result<Vec<f32>> {
    if p.is_empty() {
        return Err(Error::Domain("simplex vector must be non-empty"));
    }
    if p.iter().any(|&x| !x.is_finite()) {
        return Err(Error::Domain("vector contains non-finite values"));
    }
    if p.iter().any(|&x| x < 0.0) {
        return Err(Error::Domain("vector must be nonnegative to normalize"));
    }
    let s: f32 = p.iter().sum();
    if !(s > 0.0) {
        return Err(Error::Domain("vector must have positive total mass"));
    }
    Ok(p.iter().map(|&x| x / s).collect())
}

/// Linear interpolant on the simplex: `p(t) = (1-t)p0 + t p1`.
///
/// This does not enforce normalization; if `p0` and `p1` are on the simplex,
/// then `p(t)` is on the simplex for `t in [0,1]`.
pub fn simplex_lerp(p0: &[f32], p1: &[f32], t: f32) -> Result<Vec<f32>> {
    if p0.len() != p1.len() {
        return Err(Error::Shape("p0 and p1 must have same length"));
    }
    if !(0.0..=1.0).contains(&t) || !t.is_finite() {
        return Err(Error::Domain("t must be in [0,1] and finite"));
    }
    let mut out = Vec::with_capacity(p0.len());
    for i in 0..p0.len() {
        out.push((1.0 - t) * p0[i] + t * p1[i]);
    }
    Ok(out)
}

/// Sample a Dirichlet distribution with parameters `alpha` using Gamma draws.
///
/// Returns a simplex vector of the same length as `alpha`.
pub fn sample_dirichlet(alpha: &[f32], rng: &mut impl rand::Rng) -> Result<Vec<f32>> {
    if alpha.is_empty() {
        return Err(Error::Domain("alpha must be non-empty"));
    }
    if alpha.iter().any(|&a| !(a > 0.0) || !a.is_finite()) {
        return Err(Error::Domain("Dirichlet alpha must be positive and finite"));
    }

    use rand_distr::{Distribution, Gamma};
    let mut xs = vec![0.0f32; alpha.len()];
    let mut s: f64 = 0.0;
    for (i, &a) in alpha.iter().enumerate() {
        let g = Gamma::new(a as f64, 1.0).map_err(|_| Error::Domain("invalid Gamma params"))?;
        let x: f64 = g.sample(rng);
        xs[i] = x as f32;
        s += x;
    }
    if !(s > 0.0) {
        return Err(Error::Domain("Dirichlet sampling produced zero total mass"));
    }
    for v in &mut xs {
        *v = (*v as f64 / s) as f32;
    }
    Ok(xs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn dirichlet_samples_are_on_simplex() {
        let alpha = vec![0.7f32, 1.3, 2.1, 0.5];
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let p = sample_dirichlet(&alpha, &mut rng).unwrap();
        validate_simplex(&p, 1e-5).unwrap();
        assert!(p.iter().all(|&x| x >= 0.0));
    }
}
