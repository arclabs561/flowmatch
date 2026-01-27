//! Rectified Flow Matching (RFM) helpers.
//!
//! In the FM literature, “rectification” is largely about **the coupling** \(\pi(x_0,x_1)\):
//! choosing (or improving) which base samples are paired with which target samples so that
//! trajectories become straighter / less “curly”.
//!
//! This module provides small, testable coupling utilities built on top of `wass` (OT).

use crate::{Error, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[inline]
fn l2_squared(a: &ArrayView1<'_, f32>, b: &ArrayView1<'_, f32>) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[inline]
fn l2(a: &ArrayView1<'_, f32>, b: &ArrayView1<'_, f32>) -> f32 {
    l2_squared(a, b).sqrt()
}

fn euclidean_cost_matrix_from_views(
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
) -> Result<Array2<f32>> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(Error::Shape("x and y must have same number of rows"));
    }
    if x.ncols() != y.ncols() {
        return Err(Error::Shape("x and y must have same dimension"));
    }
    let mut cost = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        let xi = x.row(i);
        for j in 0..n {
            let yj = y.row(j);
            // Match `wass::euclidean_cost_matrix`: this is Euclidean distance (sqrt), not squared.
            // This matters for Sinkhorn and for any exp(-C / ε) transformation.
            cost[[i, j]] = l2(&xi, &yj);
        }
    }
    Ok(cost)
}

/// Build a **greedy one-to-one matching** (a permutation) from a nonnegative `n×n` weight matrix.
///
/// This is not the Hungarian algorithm; it’s a simple deterministic approximation:
/// take the largest remaining entry, match that row/col, repeat.
pub fn greedy_bipartite_match_from_weights(w: &ArrayView2<f32>) -> Result<Vec<usize>> {
    let n = w.nrows();
    if w.ncols() != n {
        return Err(Error::Shape("weight matrix must be square"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if w.iter().any(|&x| x.is_nan()) {
        return Err(Error::Domain("weight matrix contains NaN"));
    }
    if w.iter().any(|&x| x < 0.0) {
        return Err(Error::Domain("weight matrix must be nonnegative"));
    }

    // Collect all edges, sort by weight desc, then greedily assign.
    let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            edges.push((i, j, w[[i, j]]));
        }
    }
    edges.sort_by(|a, b| b.2.total_cmp(&a.2));

    let mut matched_row = vec![false; n];
    let mut matched_col = vec![false; n];
    let mut perm = vec![usize::MAX; n];
    let mut remaining = n;

    for (i, j, _wij) in edges {
        if matched_row[i] || matched_col[j] {
            continue;
        }
        matched_row[i] = true;
        matched_col[j] = true;
        perm[i] = j;
        remaining -= 1;
        if remaining == 0 {
            break;
        }
    }

    if perm.iter().any(|&j| j == usize::MAX) {
        return Err(Error::Domain("failed to construct a full matching"));
    }
    Ok(perm)
}

/// Compute a minibatch OT coupling between two equal-sized point clouds `x` and `y`,
/// then return a greedy one-to-one matching derived from the transport plan.
///
/// - `x`: `n×d`
/// - `y`: `n×d`
///
/// This is a practical “minibatch OT pairing” primitive used in rectified/OT flow matching
/// papers (pair within batch, then train on straight lines between paired points).
pub fn minibatch_ot_greedy_pairing(
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    reg: f32,
    max_iter: usize,
    tol: f32,
) -> Result<Vec<usize>> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(Error::Shape("x and y must have same number of rows"));
    }
    if x.ncols() != y.ncols() {
        return Err(Error::Shape("x and y must have same dimension"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if !(reg > 0.0) || !reg.is_finite() {
        return Err(Error::Domain("reg must be positive and finite"));
    }
    if max_iter == 0 {
        return Err(Error::Domain("max_iter must be >= 1"));
    }
    if !(tol > 0.0) || !tol.is_finite() {
        return Err(Error::Domain("tol must be positive and finite"));
    }

    // Avoid allocating/copying x/y (views) just to build the cost matrix.
    // This matters a lot for non-Sinkhorn pairing paths where cost construction dominates.
    let cost = euclidean_cost_matrix_from_views(x, y)?;

    // Uniform marginals.
    let a = Array1::<f32>::from_elem(n, 1.0 / n as f32);
    let b = Array1::<f32>::from_elem(n, 1.0 / n as f32);

    let (plan, _dist, _iters) =
        wass::sinkhorn_log_with_convergence(&a, &b, &cost, reg, max_iter, tol)
            .map_err(|_| Error::Domain("sinkhorn coupling did not converge"))?;

    greedy_bipartite_match_from_weights(&plan.view())
}

/// Partial Sinkhorn pairing:
/// - compute a Sinkhorn plan between `x` and `y`
/// - compute each row's expected transport cost under the plan
/// - select the `keep_frac` rows with the **lowest** expected cost ("easy" rows)
/// - enforce one-to-one matching for those rows using the Sinkhorn plan as weights
/// - for remaining rows, fall back to per-row nearest neighbor (duplicates allowed)
///
/// This keeps Sinkhorn's global coupling signal, but avoids the strict "use every column" constraint
/// in the final assignment, which can otherwise force minibatch outliers to be used.
pub fn minibatch_ot_selective_pairing(
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    reg: f32,
    max_iter: usize,
    tol: f32,
    keep_frac: f32,
) -> Result<Vec<usize>> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(Error::Shape("x and y must have same number of rows"));
    }
    if x.ncols() != y.ncols() {
        return Err(Error::Shape("x and y must have same dimension"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if !(reg > 0.0) || !reg.is_finite() {
        return Err(Error::Domain("reg must be positive and finite"));
    }
    if max_iter == 0 {
        return Err(Error::Domain("max_iter must be >= 1"));
    }
    if !(tol > 0.0) || !tol.is_finite() {
        return Err(Error::Domain("tol must be positive and finite"));
    }
    if !(keep_frac > 0.0) || !keep_frac.is_finite() {
        return Err(Error::Domain("keep_frac must be positive and finite"));
    }
    let keep_frac = keep_frac.min(1.0);

    let cost = euclidean_cost_matrix_from_views(x, y)?;
    let a = Array1::<f32>::from_elem(n, 1.0 / n as f32);
    let b = Array1::<f32>::from_elem(n, 1.0 / n as f32);
    let (plan, _dist, _iters) =
        wass::sinkhorn_log_with_convergence(&a, &b, &cost, reg, max_iter, tol)
            .map_err(|_| Error::Domain("sinkhorn coupling did not converge"))?;

    // Per-row: expected cost under Sinkhorn plan, and an argmin-by-cost fallback.
    let mut row_exp_cost = vec![0.0f32; n];
    let mut row_nn = vec![0usize; n];
    for i in 0..n {
        // expected cost
        let mut e = 0.0f32;
        // nearest neighbor by cost (squared cost is fine for argmin; cost matrix is sqrt-L2 but monotone)
        let mut best_j = 0usize;
        let mut best_c = f32::INFINITY;
        for j in 0..n {
            e += plan[[i, j]] * cost[[i, j]];
            let c = cost[[i, j]];
            if c < best_c {
                best_c = c;
                best_j = j;
            }
        }
        row_exp_cost[i] = e;
        row_nn[i] = best_j;
    }

    // Choose the top-k rows by lowest expected cost (easiest rows).
    let keep = ((keep_frac * n as f32).round() as usize).clamp(1, n);
    let mut rows: Vec<usize> = (0..n).collect();
    rows.sort_by(|&i, &j| row_exp_cost[i].total_cmp(&row_exp_cost[j])); // asc
    let selected = &rows[..keep];

    // Assign selected rows one-to-one, greedily by weight.
    let mut selected_sorted = selected.to_vec();
    selected_sorted.sort_by(|&i, &j| row_exp_cost[i].total_cmp(&row_exp_cost[j])); // asc

    let mut used_col = vec![false; n];
    let mut perm = vec![usize::MAX; n];

    for &i in &selected_sorted {
        let mut best_j = usize::MAX;
        let mut best_w = -1.0f32;
        for j in 0..n {
            if used_col[j] {
                continue;
            }
            let w = plan[[i, j]];
            if w > best_w {
                best_w = w;
                best_j = j;
            }
        }
        if best_j == usize::MAX {
            return Err(Error::Domain("failed to assign columns for selected rows"));
        }
        used_col[best_j] = true;
        perm[i] = best_j;
    }

    // Remaining rows: nearest neighbor by cost (duplicates allowed).
    for i in 0..n {
        if perm[i] != usize::MAX {
            continue;
        }
        perm[i] = row_nn[i];
    }

    debug_assert!(perm.iter().all(|&j| j < n));
    Ok(perm)
}

/// Fast minibatch pairing: greedy row-wise nearest neighbor on the Euclidean cost matrix.
pub fn minibatch_rowwise_nearest_pairing(
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
) -> Result<Vec<usize>> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(Error::Shape("x and y must have same number of rows"));
    }
    if x.ncols() != y.ncols() {
        return Err(Error::Shape("x and y must have same dimension"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    // Avoid materializing the full n×n cost matrix:
    // for each row i, pick min unused column by computing distances on the fly.
    if x.iter().any(|&v| !v.is_finite()) || y.iter().any(|&v| !v.is_finite()) {
        return Err(Error::Domain("x/y contain NaN/Inf"));
    }
    let mut used = vec![false; n];
    let mut perm = vec![usize::MAX; n];
    for i in 0..n {
        let xi = x.row(i);
        let mut best_j = usize::MAX;
        let mut best = f32::INFINITY;
        for j in 0..n {
            if used[j] {
                continue;
            }
            let yj = y.row(j);
            // Use squared distance for speed. Argmin is identical to Euclidean distance.
            let c = l2_squared(&xi, &yj);
            if c < best {
                best = c;
                best_j = j;
            }
        }
        if best_j == usize::MAX {
            return Err(Error::Domain("failed to construct a full matching"));
        }
        used[best_j] = true;
        perm[i] = best_j;
    }
    Ok(perm)
}

/// Partial pairing heuristic:
/// - compute each row's nearest-neighbor distance
/// - take the best `keep_frac` rows (lowest distances) and match them one-to-one (unique columns)
/// - for the remaining rows, assign the nearest column (duplicates allowed)
///
/// This is a pragmatic stand-in for "partial OT" in minibatches: it avoids forcing every column
/// to be used, which can create huge, misleading displacements when a minibatch contains a rare
/// outlier target.
pub fn minibatch_partial_rowwise_pairing(
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    keep_frac: f32,
) -> Result<Vec<usize>> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(Error::Shape("x and y must have same number of rows"));
    }
    if x.ncols() != y.ncols() {
        return Err(Error::Shape("x and y must have same dimension"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if !(keep_frac > 0.0) || !keep_frac.is_finite() {
        return Err(Error::Domain("keep_frac must be positive and finite"));
    }
    let keep_frac = keep_frac.min(1.0);

    if x.iter().any(|&v| !v.is_finite()) || y.iter().any(|&v| !v.is_finite()) {
        return Err(Error::Domain("x/y contain NaN/Inf"));
    }

    // For each row, compute (best_j, best_cost_sq).
    let mut best: Vec<(usize, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        let xi = x.row(i);
        let mut best_j = 0usize;
        let mut best_c = f32::INFINITY;
        for j in 0..n {
            let yj = y.row(j);
            let c = l2_squared(&xi, &yj);
            if c < best_c {
                best_c = c;
                best_j = j;
            }
        }
        best.push((best_j, best_c));
    }

    // Select top-k rows by smallest nearest-neighbor cost.
    let keep = ((keep_frac * n as f32).round() as usize).clamp(1, n);
    let mut rows: Vec<usize> = (0..n).collect();
    rows.sort_by(|&i, &j| best[i].1.total_cmp(&best[j].1)); // ascending by cost
    let selected = &rows[..keep];

    // One-to-one matching for selected rows, greedy by their best cost.
    let mut selected_sorted = selected.to_vec();
    selected_sorted.sort_by(|&i, &j| best[i].1.total_cmp(&best[j].1));

    let mut used_col = vec![false; n];
    let mut perm = vec![usize::MAX; n];

    for &i in &selected_sorted {
        let xi = x.row(i);
        // Choose best unused col (not necessarily the argmin col if it's already used).
        let mut best_j = usize::MAX;
        let mut best_c = f32::INFINITY;
        for j in 0..n {
            if used_col[j] {
                continue;
            }
            let yj = y.row(j);
            let c = l2_squared(&xi, &yj);
            if c < best_c {
                best_c = c;
                best_j = j;
            }
        }
        if best_j == usize::MAX {
            return Err(Error::Domain("failed to assign columns for selected rows"));
        }
        used_col[best_j] = true;
        perm[i] = best_j;
    }

    // Remaining rows: nearest neighbor (duplicates allowed).
    for i in 0..n {
        if perm[i] != usize::MAX {
            continue;
        }
        perm[i] = best[i].0;
    }

    debug_assert!(perm.iter().all(|&j| j < n));
    Ok(perm)
}

/// Fast minibatch pairing: convert costs to weights via `exp(-cost / temp)` then greedy match.
pub fn minibatch_exp_greedy_pairing(
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    temp: f32,
) -> Result<Vec<usize>> {
    let n = x.nrows();
    if y.nrows() != n {
        return Err(Error::Shape("x and y must have same number of rows"));
    }
    if x.ncols() != y.ncols() {
        return Err(Error::Shape("x and y must have same dimension"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if !(temp > 0.0) || !temp.is_finite() {
        return Err(Error::Domain("temp must be positive and finite"));
    }

    if x.iter().any(|&v| !v.is_finite()) || y.iter().any(|&v| !v.is_finite()) {
        return Err(Error::Domain("x/y contain NaN/Inf"));
    }

    // Avoid materializing cost and weight matrices. We still build an n² edge list to sort.
    let d = x.ncols();
    let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(n * n);
    for i in 0..n {
        let xi = x.row(i);
        for j in 0..n {
            let yj = y.row(j);
            // Match Sinkhorn’s notion of cost here (Euclidean distance, not squared).
            let mut dist_sq = 0.0f32;
            for k in 0..d {
                let dd = xi[k] - yj[k];
                dist_sq += dd * dd;
            }
            let dist = dist_sq.sqrt();
            edges.push((i, j, (-dist / temp).exp()));
        }
    }
    edges.sort_by(|a, b| b.2.total_cmp(&a.2));

    let mut matched_row = vec![false; n];
    let mut matched_col = vec![false; n];
    let mut perm = vec![usize::MAX; n];
    let mut remaining = n;

    for (i, j, _wij) in edges {
        if matched_row[i] || matched_col[j] {
            continue;
        }
        matched_row[i] = true;
        matched_col[j] = true;
        perm[i] = j;
        remaining -= 1;
        if remaining == 0 {
            break;
        }
    }

    if perm.iter().any(|&j| j == usize::MAX) {
        return Err(Error::Domain("failed to construct a full matching"));
    }
    Ok(perm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use proptest::prelude::*;

    #[test]
    fn greedy_matching_is_a_permutation() {
        let w = array![[0.9, 0.1, 0.0], [0.2, 0.8, 0.1], [0.0, 0.1, 0.7]];
        let p = greedy_bipartite_match_from_weights(&w.view()).unwrap();
        assert_eq!(p.len(), 3);
        let mut seen = vec![false; 3];
        for &j in &p {
            assert!(j < 3);
            assert!(!seen[j]);
            seen[j] = true;
        }
    }

    #[test]
    fn minibatch_ot_pairing_is_deterministic_and_a_permutation() {
        let x = array![[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![[0.0f32, 0.0], [1.0, 0.1], [0.1, 1.0], [1.0, 1.0]];

        let p1 = minibatch_ot_greedy_pairing(&x.view(), &y.view(), 0.2, 5000, 2e-3).unwrap();
        let p2 = minibatch_ot_greedy_pairing(&x.view(), &y.view(), 0.2, 5000, 2e-3).unwrap();
        assert_eq!(p1, p2);

        let mut seen = vec![false; p1.len()];
        for &j in &p1 {
            assert!(j < p1.len());
            assert!(!seen[j]);
            seen[j] = true;
        }
    }

    fn is_permutation(p: &[usize]) -> bool {
        let n = p.len();
        let mut seen = vec![false; n];
        for &j in p {
            if j >= n || seen[j] {
                return false;
            }
            seen[j] = true;
        }
        true
    }

    fn is_in_range(p: &[usize]) -> bool {
        let n = p.len();
        p.iter().all(|&j| j < n)
    }

    fn selected_rows_by_nn_cost_sq(
        x: &ArrayView2<f32>,
        y: &ArrayView2<f32>,
        keep_frac: f32,
    ) -> Vec<usize> {
        let n = x.nrows();
        let keep = ((keep_frac.min(1.0) * n as f32).round() as usize).clamp(1, n);
        let mut best_cost: Vec<(usize, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            let xi = x.row(i);
            let mut best = f32::INFINITY;
            for j in 0..n {
                let yj = y.row(j);
                let c = l2_squared(&xi, &yj);
                if c < best {
                    best = c;
                }
            }
            best_cost.push((i, best));
        }
        best_cost.sort_by(|a, b| a.1.total_cmp(&b.1)); // asc
        best_cost.into_iter().take(keep).map(|(i, _)| i).collect()
    }

    fn rowwise_nearest_pairing_sqrt_reference(
        x: &ArrayView2<f32>,
        y: &ArrayView2<f32>,
    ) -> Result<Vec<usize>> {
        let n = x.nrows();
        if y.nrows() != n {
            return Err(Error::Shape("x and y must have same number of rows"));
        }
        if x.ncols() != y.ncols() {
            return Err(Error::Shape("x and y must have same dimension"));
        }
        if x.iter().any(|&v| !v.is_finite()) || y.iter().any(|&v| !v.is_finite()) {
            return Err(Error::Domain("x/y contain NaN/Inf"));
        }
        let mut used = vec![false; n];
        let mut perm = vec![usize::MAX; n];
        for i in 0..n {
            let xi = x.row(i);
            let mut best_j = usize::MAX;
            let mut best = f32::INFINITY;
            for j in 0..n {
                if used[j] {
                    continue;
                }
                let yj = y.row(j);
                let c = l2(&xi, &yj);
                if c < best {
                    best = c;
                    best_j = j;
                }
            }
            if best_j == usize::MAX {
                return Err(Error::Domain("failed to construct a full matching"));
            }
            used[best_j] = true;
            perm[i] = best_j;
        }
        Ok(perm)
    }

    // This is the exact seam we broke previously: we must match `wass::euclidean_cost_matrix`
    // semantics (it uses Euclidean distance, i.e. sqrt of squared distance).
    proptest! {
        #[test]
        fn prop_cost_matrix_matches_wass(
            n in 1usize..8,
            d in 1usize..8,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::{Distribution, StandardNormal};

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut x = Array2::<f32>::zeros((n, d));
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    x[[i, k]] = StandardNormal.sample(&mut rng);
                    y[[i, k]] = StandardNormal.sample(&mut rng);
                }
            }

            let cost_wass = wass::euclidean_cost_matrix(&x, &y);
            let cost_ours = euclidean_cost_matrix_from_views(&x.view(), &y.view()).unwrap();

            prop_assert_eq!(cost_wass.shape(), cost_ours.shape());
            for i in 0..n {
                for j in 0..n {
                    let a = cost_wass[[i, j]];
                    let b = cost_ours[[i, j]];
                    prop_assert!((a - b).abs() <= 1e-6, "mismatch at ({i},{j}): wass={a} ours={b}");
                }
            }
        }
    }

    // Another seam where it’s easy to lie: "squared vs sqrt doesn't matter".
    // It *doesn't* matter for argmin, but it *does* matter for exp(-cost / temp) weights.
    proptest! {
        #[test]
        fn prop_argmin_same_for_sq_vs_sqrt(
            n in 1usize..16,
            d in 1usize..16,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::{Distribution, StandardNormal};

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut xs: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut ys: Vec<Vec<f32>> = Vec::with_capacity(n);
            for _ in 0..n {
                let mut v = vec![0.0f32; d];
                let mut w = vec![0.0f32; d];
                for k in 0..d {
                    v[k] = StandardNormal.sample(&mut rng);
                    w[k] = StandardNormal.sample(&mut rng);
                }
                xs.push(v);
                ys.push(w);
            }

            let x0 = ndarray::ArrayView1::from(&xs[0]);
            let mut best_sq = f32::INFINITY;
            let mut best_sq_j = 0usize;
            let mut best_eu = f32::INFINITY;
            let mut best_eu_j = 0usize;
            for (j, yj_vec) in ys.iter().enumerate() {
                let yj = ndarray::ArrayView1::from(yj_vec);
                let dsq = l2_squared(&x0, &yj);
                let deu = dsq.sqrt();
                if dsq < best_sq {
                    best_sq = dsq;
                    best_sq_j = j;
                }
                if deu < best_eu {
                    best_eu = deu;
                    best_eu_j = j;
                }
            }

            prop_assert_eq!(best_sq_j, best_eu_j);
        }
    }

    // "Exp-greedy" must mean: w_ij = exp(-wass_cost_ij / temp), then greedy matching.
    proptest! {
        #[test]
        fn prop_exp_greedy_matches_weight_matrix_definition(
            n in 1usize..7,
            d in 1usize..8,
            temp in 0.05f32..1.0f32,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::{Distribution, StandardNormal};

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut x = Array2::<f32>::zeros((n, d));
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    x[[i, k]] = StandardNormal.sample(&mut rng);
                    y[[i, k]] = StandardNormal.sample(&mut rng);
                }
            }

            let perm_fast = minibatch_exp_greedy_pairing(&x.view(), &y.view(), temp).unwrap();

            let cost = wass::euclidean_cost_matrix(&x, &y);
            let mut w = cost.clone();
            for i in 0..n {
                for j in 0..n {
                    w[[i, j]] = (-cost[[i, j]] / temp).exp();
                }
            }
            let perm_def = greedy_bipartite_match_from_weights(&w.view()).unwrap();

            prop_assert!(is_permutation(&perm_fast));
            prop_assert_eq!(perm_fast, perm_def);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_partial_rowwise_in_range_and_deterministic(
            n in 1usize..32,
            d in 1usize..16,
            keep_frac in 0.05f32..1.0f32,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::{Distribution, StandardNormal};

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut x = Array2::<f32>::zeros((n, d));
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    x[[i, k]] = StandardNormal.sample(&mut rng);
                    y[[i, k]] = StandardNormal.sample(&mut rng);
                }
            }

            let p1 = minibatch_partial_rowwise_pairing(&x.view(), &y.view(), keep_frac).unwrap();
            let p2 = minibatch_partial_rowwise_pairing(&x.view(), &y.view(), keep_frac).unwrap();
            prop_assert_eq!(&p1, &p2);
            prop_assert_eq!(p1.len(), n);
            prop_assert!(is_in_range(&p1));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 16,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_sinkhorn_selective_in_range_and_deterministic(
            n in 1usize..10,
            d in 1usize..10,
            keep_frac in 0.1f32..1.0f32,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::{Distribution, StandardNormal};

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut x = Array2::<f32>::zeros((n, d));
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    x[[i, k]] = StandardNormal.sample(&mut rng);
                    y[[i, k]] = StandardNormal.sample(&mut rng);
                }
            }

            // Use stable parameters to avoid proptest flake.
            let reg = 0.2;
            let max_iter = 5_000;
            let tol = 2e-3;

            let p1 = minibatch_ot_selective_pairing(&x.view(), &y.view(), reg, max_iter, tol, keep_frac).unwrap();
            let p2 = minibatch_ot_selective_pairing(&x.view(), &y.view(), reg, max_iter, tol, keep_frac).unwrap();
            prop_assert_eq!(&p1, &p2);
            prop_assert_eq!(p1.len(), n);
            prop_assert!(is_in_range(&p1));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_partial_rowwise_keep_frac_enforces_unique_cols_on_selected_rows(
            n in 2usize..32,
            d in 1usize..16,
            keep_frac in 0.05f32..1.0f32,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::{Distribution, StandardNormal};

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut x = Array2::<f32>::zeros((n, d));
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    x[[i, k]] = StandardNormal.sample(&mut rng);
                    y[[i, k]] = StandardNormal.sample(&mut rng);
                }
            }

            let perm = minibatch_partial_rowwise_pairing(&x.view(), &y.view(), keep_frac).unwrap();
            prop_assert_eq!(perm.len(), n);
            prop_assert!(is_in_range(&perm));

            let selected = selected_rows_by_nn_cost_sq(&x.view(), &y.view(), keep_frac);
            let mut seen = std::collections::HashSet::<usize>::new();
            for &i in &selected {
                let j = perm[i];
                prop_assert!(seen.insert(j), "expected unique columns for selected rows; duplicate col {j}");
            }
        }
    }

    #[test]
    fn keep_frac_monotone_outlier_usage_partial_rowwise_and_sinkhorn_selective() {
        // Deterministic "outlier forcing" regime: last y is far away from everything.
        let n = 32usize;
        let d = 8usize;
        let mut x = Array2::<f32>::zeros((n, d));
        let mut y = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for k in 0..d {
                x[[i, k]] = (i as f32) * 0.01 + (k as f32) * 0.001;
                y[[i, k]] = (i as f32) * 0.01 + (k as f32) * 0.001;
            }
        }
        for k in 0..d {
            y[[n - 1, k]] = 1_000.0;
        }

        let count_outlier = |perm: &[usize]| perm.iter().filter(|&&j| j == n - 1).count();

        // Partial rowwise: keep_frac=1 should force using outlier (full one-to-one),
        // keep_frac=0.5 should avoid it (NN fallback never picks an extreme outlier).
        let p_full = minibatch_partial_rowwise_pairing(&x.view(), &y.view(), 1.0).unwrap();
        let p_half = minibatch_partial_rowwise_pairing(&x.view(), &y.view(), 0.5).unwrap();
        assert_eq!(count_outlier(&p_full), 1);
        assert_eq!(count_outlier(&p_half), 0);

        // Sinkhorn selective: same expectation.
        let s_full =
            minibatch_ot_selective_pairing(&x.view(), &y.view(), 0.2, 5_000, 2e-3, 1.0).unwrap();
        let s_half =
            minibatch_ot_selective_pairing(&x.view(), &y.view(), 0.2, 5_000, 2e-3, 0.5).unwrap();
        assert_eq!(count_outlier(&s_full), 1);
        assert_eq!(count_outlier(&s_half), 0);
    }

    // If Sinkhorn pairing succeeds, it should always produce a permutation.
    // (We don't assert success for arbitrary random inputs + iteration limits.)
    proptest! {
        #[test]
        fn prop_sinkhorn_pairing_ok_implies_permutation(
            n in 2usize..10,
            d in 1usize..8,
            reg in 0.05f32..0.5f32,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::{Distribution, StandardNormal};

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut x = Array2::<f32>::zeros((n, d));
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    x[[i, k]] = StandardNormal.sample(&mut rng);
                    y[[i, k]] = StandardNormal.sample(&mut rng);
                }
            }

            if let Ok(p) = minibatch_ot_greedy_pairing(&x.view(), &y.view(), reg, 500, 2e-2) {
                prop_assert_eq!(p.len(), n);
                prop_assert!(is_permutation(&p));
            }
        }
    }

    // OT coupling invariance (paper nuance): adding separable row/col shifts to the cost matrix
    // must not change the optimal coupling (up to Sinkhorn scaling), hence should not change the
    // greedy matching extracted from the plan.
    //
    // This would have caught our earlier “we accidentally changed the ground cost” regression.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_sinkhorn_plan_approximately_invariant_to_rowcol_shifts(
            n in 2usize..7,
            d in 1usize..8,
            reg in 0.10f32..0.5f32,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::{Distribution, StandardNormal};

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut x = Array2::<f32>::zeros((n, d));
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    x[[i, k]] = StandardNormal.sample(&mut rng);
                    y[[i, k]] = StandardNormal.sample(&mut rng);
                }
            }

            let cost = wass::euclidean_cost_matrix(&x, &y);

            // Build separable shifts c_i + d_j (small magnitude).
            let mut row = vec![0.0f32; n];
            let mut col = vec![0.0f32; n];
            for i in 0..n {
                let r: f32 = StandardNormal.sample(&mut rng);
                let c: f32 = StandardNormal.sample(&mut rng);
                row[i] = r * 0.1;
                col[i] = c * 0.1;
            }
            let mut cost2 = cost.clone();
            for i in 0..n {
                for j in 0..n {
                    cost2[[i, j]] = cost[[i, j]] + row[i] + col[j];
                }
            }

            let a = Array1::<f32>::from_elem(n, 1.0 / n as f32);
            let b = Array1::<f32>::from_elem(n, 1.0 / n as f32);

            // With finite iterations + tolerances, we only get approximate invariance.
            // Compare the transport plans directly; greedy match can differ under near-ties.
            let (p1, _d1, _it1) = wass::sinkhorn_log_with_convergence(&a, &b, &cost, reg, 20_000, 1e-4).unwrap();
            let (p2, _d2, _it2) = wass::sinkhorn_log_with_convergence(&a, &b, &cost2, reg, 20_000, 1e-4).unwrap();

            let mut max_abs = 0.0f32;
            for i in 0..n {
                for j in 0..n {
                    let d = (p1[[i, j]] - p2[[i, j]]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
            }

            // This tolerance is intentionally loose: it’s a regression guard, not a numerical proof.
            // If this trips, we likely changed cost semantics or the Sinkhorn implementation.
            prop_assert!(max_abs <= 5e-3, "expected near-invariant plan; max_abs={max_abs}");
        }
    }

    // This is a common “we thought it didn’t matter” failure mode:
    // squared vs sqrt distances. For per-row argmin it shouldn't matter, but because we build
    // a greedy one-to-one permutation, it’s worth pinning down as a property.
    proptest! {
        #[test]
        fn prop_rowwise_pairing_invariant_to_sqrt(
            n in 2usize..32,
            d in 1usize..16,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::{Distribution, StandardNormal};

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut x = Array2::<f32>::zeros((n, d));
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    x[[i, k]] = StandardNormal.sample(&mut rng);
                    y[[i, k]] = StandardNormal.sample(&mut rng);
                }
            }

            let p_sq = minibatch_rowwise_nearest_pairing(&x.view(), &y.view()).unwrap();
            let p_sqrt = rowwise_nearest_pairing_sqrt_reference(&x.view(), &y.view()).unwrap();
            prop_assert!(is_permutation(&p_sq));
            prop_assert_eq!(p_sq, p_sqrt);
        }
    }

    #[test]
    fn partial_rowwise_avoids_forcing_outlier_column() {
        let n = 16usize;
        let d = 4usize;
        let mut x = Array2::<f32>::zeros((n, d));
        let mut y = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for k in 0..d {
                x[[i, k]] = i as f32 * 0.01;
                y[[i, k]] = i as f32 * 0.01;
            }
        }
        // Make last y an extreme outlier.
        for k in 0..d {
            y[[n - 1, k]] = 1_000.0;
        }

        let full = minibatch_rowwise_nearest_pairing(&x.view(), &y.view()).unwrap();
        let partial = minibatch_partial_rowwise_pairing(&x.view(), &y.view(), 0.8).unwrap();

        // Full one-to-one must use every column, thus must include outlier column exactly once.
        assert_eq!(full.iter().filter(|&&j| j == n - 1).count(), 1);
        // Partial should not be forced to use the outlier column.
        assert_eq!(partial.iter().filter(|&&j| j == n - 1).count(), 0);
    }

    #[test]
    fn sinkhorn_selective_avoids_forcing_outlier_column() {
        let n = 16usize;
        let d = 4usize;
        let mut x = Array2::<f32>::zeros((n, d));
        let mut y = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for k in 0..d {
                x[[i, k]] = i as f32 * 0.01;
                y[[i, k]] = i as f32 * 0.01;
            }
        }
        for k in 0..d {
            y[[n - 1, k]] = 1_000.0;
        }

        let full = minibatch_ot_greedy_pairing(&x.view(), &y.view(), 0.2, 2_000, 1e-4).unwrap();
        let sel =
            minibatch_ot_selective_pairing(&x.view(), &y.view(), 0.2, 2_000, 1e-4, 0.8).unwrap();

        // Full one-to-one must use every column, thus must include outlier column exactly once.
        assert_eq!(full.iter().filter(|&&j| j == n - 1).count(), 1);
        // Selective pairing should not be forced to use the outlier column.
        assert_eq!(sel.iter().filter(|&&j| j == n - 1).count(), 0);
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_pairing_apis_error_on_shape_mismatch_or_nan(
            n in 1usize..16,
            d in 1usize..16,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::{Distribution, StandardNormal};

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut x = Array2::<f32>::zeros((n, d));
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    x[[i, k]] = StandardNormal.sample(&mut rng);
                    y[[i, k]] = StandardNormal.sample(&mut rng);
                }
            }

            // Shape mismatch (different dims) should error.
            let y_bad = Array2::<f32>::zeros((n, d + 1));
            prop_assert!(minibatch_rowwise_nearest_pairing(&x.view(), &y_bad.view()).is_err());
            prop_assert!(minibatch_partial_rowwise_pairing(&x.view(), &y_bad.view(), 0.8).is_err());
            prop_assert!(minibatch_exp_greedy_pairing(&x.view(), &y_bad.view(), 0.2).is_err());
            prop_assert!(minibatch_ot_greedy_pairing(&x.view(), &y_bad.view(), 0.2, 100, 1e-2).is_err());
            prop_assert!(minibatch_ot_selective_pairing(&x.view(), &y_bad.view(), 0.2, 100, 1e-2, 0.8).is_err());

            // NaN should error for fast pairings (and should not panic for Sinkhorn pairings).
            x[[0, 0]] = f32::NAN;
            prop_assert!(minibatch_rowwise_nearest_pairing(&x.view(), &y.view()).is_err());
            prop_assert!(minibatch_partial_rowwise_pairing(&x.view(), &y.view(), 0.8).is_err());
            prop_assert!(minibatch_exp_greedy_pairing(&x.view(), &y.view(), 0.2).is_err());
            prop_assert!(minibatch_ot_greedy_pairing(&x.view(), &y.view(), 0.2, 200, 1e-2).is_err());
            prop_assert!(minibatch_ot_selective_pairing(&x.view(), &y.view(), 0.2, 200, 1e-2, 0.8).is_err());
        }
    }
}
