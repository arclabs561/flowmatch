//! Semidiscrete flow matching (SD-FM), minimal version.
//!
//! This module wires together:
//! - a semidiscrete coupling (potentials + assignments) from `wass::semidiscrete`
//! - a simple conditional vector field
//! - an SGD loop for the flow-matching regression objective
//!
//! It is intentionally not a “research-complete” reproduction. The contract is:
//! - deterministic with a seed,
//! - measurable improvement on a small synthetic regime (via an e2e test),
//! - no hidden assumptions.

use crate::linear::LinearCondField;
use crate::ode::{integrate_fixed, OdeMethod};
use crate::rfm::{
    minibatch_exp_greedy_pairing, minibatch_ot_greedy_pairing, minibatch_ot_selective_pairing,
    minibatch_partial_rowwise_pairing, minibatch_rowwise_nearest_pairing,
};
use crate::{Error, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use wass::semidiscrete::{
    assign_hard_from_scores, fit_potentials_sgd_neg_dot, scores_neg_dot, SemidiscreteSgdConfig,
};

fn sample_categorical_from_probs(probs: &ArrayView1<f32>, rng: &mut impl rand::Rng) -> usize {
    debug_assert!(probs.len() > 0);
    debug_assert!(probs.iter().all(|&x| x >= 0.0 && x.is_finite()));
    // Note: even if `probs.sum()` is very close to 1, float roundoff can leave the cumulative
    // sum slightly below 1.0. We therefore fall back to the last index instead of biasing to 0.
    let u: f32 = rng.random();
    let mut acc = 0.0f32;
    for idx in 0..probs.len() {
        acc += probs[idx];
        if u <= acc {
            return idx;
        }
    }
    probs.len() - 1
}

/// How we sample the FM time variable `t ∈ [0,1]` during training.
///
/// Paper nuance: non-uniform (U-shaped) time sampling can materially affect few-step quality,
/// because numerical error concentrates near the boundaries. We keep this explicit and testable.
#[derive(Debug, Clone, Copy)]
pub enum TimestepSchedule {
    /// Uniform `t ~ U[0,1]`.
    Uniform,
    /// U-shaped distribution with more mass near 0 and 1.
    ///
    /// Implemented as `t = sin^2((π/2) * u)` for `u ~ U[0,1]`, which is Beta(1/2, 1/2).
    UShaped,
}

impl Default for TimestepSchedule {
    fn default() -> Self {
        Self::Uniform
    }
}

impl TimestepSchedule {
    #[inline]
    pub fn sample_t(self, rng: &mut impl rand::Rng) -> f32 {
        let u: f32 = rng.random();
        match self {
            TimestepSchedule::Uniform => u,
            TimestepSchedule::UShaped => {
                let s = (0.5 * core::f32::consts::PI * u).sin();
                s * s
            }
        }
    }
}

/// Training configuration for SD-FM.
#[derive(Debug, Clone)]
pub struct SdFmTrainConfig {
    /// SGD learning rate for the vector field.
    pub lr: f32,
    /// Number of SGD steps.
    pub steps: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Euler steps for sampling (0 means “don’t sample”).
    pub sample_steps: usize,
    /// RNG seed.
    pub seed: u64,
    /// Timestep sampling schedule.
    pub t_schedule: TimestepSchedule,
}

impl Default for SdFmTrainConfig {
    fn default() -> Self {
        Self {
            lr: 5e-3,
            steps: 2_000,
            batch_size: 512,
            sample_steps: 40,
            seed: 123,
            t_schedule: TimestepSchedule::Uniform,
        }
    }
}

/// Configuration for minibatch-OT rectified flow matching (RFM) coupling.
///
/// This config controls the **coupling** computation, not the vector field SGD.
#[derive(Debug, Clone, Copy)]
pub enum RfmMinibatchPairing {
    /// Current default: Sinkhorn OT plan + greedy matching.
    SinkhornGreedy,
    /// Partial Sinkhorn pairing: compute a Sinkhorn plan, then only enforce one-to-one matching
    /// for the most confident fraction of rows. Remaining rows fall back to per-row argmax in
    /// the Sinkhorn plan (duplicates allowed).
    ///
    /// This avoids "using every column" in the final assignment, mitigating minibatch outlier
    /// forcing while keeping Sinkhorn's global coupling signal.
    SinkhornSelective { keep_frac: f32 },
    /// Faster pairing: greedy row-wise nearest neighbor assignment on the cost matrix.
    ///
    /// This avoids Sinkhorn entirely (and is dramatically faster), but is a weaker coupling.
    RowwiseNearest,
    /// Faster pairing: convert costs to weights via `exp(-cost / temp)` then greedy matching.
    ///
    /// This is still O(n² log n) due to sorting edges, but avoids Sinkhorn iterations.
    ExpGreedy { temp: f32 },
    /// Partial pairing heuristic: only enforce one-to-one matching for the "easy" fraction of rows.
    ///
    /// For the remaining rows, we fall back to per-row nearest neighbor (allowing duplicates),
    /// which avoids forcing a match to a rare/outlier target within a minibatch.
    ///
    /// `keep_frac` is the fraction of rows to match one-to-one (clamped to \((0,1]\)).
    PartialRowwise { keep_frac: f32 },
}

#[derive(Debug, Clone)]
pub struct RfmMinibatchOtConfig {
    /// Entropic regularization `ε` for Sinkhorn (larger = easier, smaller = sharper).
    pub reg: f32,
    /// Maximum Sinkhorn iterations per minibatch coupling.
    pub max_iter: usize,
    /// Convergence tolerance for Sinkhorn marginal error.
    pub tol: f32,
    /// Pairing method used to derive a one-to-one pairing.
    pub pairing: RfmMinibatchPairing,
    /// How often to recompute the coupling (1 = every SGD step).
    ///
    /// Re-using a pairing for a few SGD steps can cut coupling time by ~`pairing_every`×.
    pub pairing_every: usize,
}

impl Default for RfmMinibatchOtConfig {
    fn default() -> Self {
        Self {
            reg: 0.2,
            max_iter: 6_000,
            tol: 2e-3,
            pairing: RfmMinibatchPairing::SinkhornGreedy,
            pairing_every: 1,
        }
    }
}

/// A trained SD-FM model (semidiscrete potentials + conditional vector field).
#[derive(Debug, Clone)]
pub struct TrainedSdFm {
    /// Discrete target support `y` (n × d).
    pub y: Array2<f32>,
    /// Target weights `b` (normalized to sum to 1).
    pub b: Array1<f32>,
    /// Semidiscrete potentials (length n, centered).
    pub g: Array1<f32>,
    /// How we assign the discrete index `j` during training/sampling.
    pub assignment: SdFmTrainAssignment,
    /// Conditional vector field.
    pub field: LinearCondField,
}

/// How to choose the discrete conditioning index `j` during FM training.
///
/// - `SemidiscretePotentials`: current default (uses `wass::semidiscrete` potentials + argmax assignment).
/// - `CategoricalFromB`: sample `j ~ b` directly (useful as a baseline / ablation).
#[derive(Debug, Clone, Copy)]
pub enum SdFmTrainAssignment {
    SemidiscretePotentials,
    CategoricalFromB,
}

impl TrainedSdFm {
    /// Like [`Self::sample`], but also returns the initial noise `x0` used for each trajectory.
    ///
    /// This exists so tests can compare against baselines without re-deriving RNG streams.
    pub fn sample_with_x0(
        &self,
        n: usize,
        seed: u64,
        steps: usize,
    ) -> Result<(Array2<f32>, Array2<f32>, Vec<usize>)> {
        self.sample_with_x0_method(n, seed, steps, OdeMethod::Euler)
    }

    /// Like [`Self::sample_with_x0`], but lets you choose the ODE integrator.
    pub fn sample_with_x0_method(
        &self,
        n: usize,
        seed: u64,
        steps: usize,
        method: OdeMethod,
    ) -> Result<(Array2<f32>, Array2<f32>, Vec<usize>)> {
        if steps == 0 {
            return Err(Error::Domain("sample steps must be >= 1"));
        }
        let d = self.y.ncols();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let dt = 1.0f32 / (steps as f32);

        let mut x0s = Array2::<f32>::zeros((n, d));
        let mut x1s = Array2::<f32>::zeros((n, d));
        let mut js: Vec<usize> = Vec::with_capacity(n);

        for i in 0..n {
            // x0
            let mut x = Array1::<f32>::zeros(d);
            for k in 0..d {
                x[k] = StandardNormal.sample(&mut rng);
                x0s[[i, k]] = x[k];
            }

            // Pick discrete index j consistently with training assignment.
            let j = match self.assignment {
                SdFmTrainAssignment::SemidiscretePotentials => {
                    let scores = scores_neg_dot(&x.view(), &self.y.view(), &self.g.view());
                    assign_hard_from_scores(&scores.view())
                }
                SdFmTrainAssignment::CategoricalFromB => {
                    // Sample j ~ b (b is normalized in TrainedSdFm).
                    sample_categorical_from_probs(&self.b.view(), &mut rng)
                }
            };
            js.push(j);
            let yj = self.y.row(j);

            // ODE integration
            let x1 = integrate_fixed(method, &x, 0.0, dt, steps, |xt, t| {
                self.field.eval(xt, t, &yj)
            });
            x = x1;

            for k in 0..d {
                x1s[[i, k]] = x[k];
            }
        }

        Ok((x0s, x1s, js))
    }

    /// Sample `n` trajectories by:
    /// - drawing `x0 ~ N(0, I)`,
    /// - choosing a discrete index `j` (according to `self.assignment`),
    /// - integrating `dx/dt = v(x,t;y_j*)` with Euler.
    ///
    /// Returns `(xs_1, js)`, where `js[i]` is the discrete index used to condition the i-th trajectory.
    pub fn sample(&self, n: usize, seed: u64, steps: usize) -> Result<(Array2<f32>, Vec<usize>)> {
        let (_x0s, x1s, js) = self.sample_with_x0(n, seed, steps)?;
        Ok((x1s, js))
    }
}

/// Train an SD-FM model on a discrete target support `y` with weights `b`.
///
/// Stages:
/// - fit semidiscrete potentials `g` so noise assignments match `b`
/// - train a conditional vector field to predict the linear-path flow target \(u = y - x_0\)
pub fn train_sd_fm_semidiscrete_linear(
    y: &ArrayView2<f32>,
    b: &ArrayView1<f32>,
    pot_cfg: &SemidiscreteSgdConfig,
    fm_cfg: &SdFmTrainConfig,
) -> Result<TrainedSdFm> {
    train_sd_fm_semidiscrete_linear_with_assignment(
        y,
        b,
        pot_cfg,
        fm_cfg,
        SdFmTrainAssignment::SemidiscretePotentials,
    )
}

/// Like [`train_sd_fm_semidiscrete_linear`], but makes the discrete assignment strategy explicit.
pub fn train_sd_fm_semidiscrete_linear_with_assignment(
    y: &ArrayView2<f32>,
    b: &ArrayView1<f32>,
    pot_cfg: &SemidiscreteSgdConfig,
    fm_cfg: &SdFmTrainConfig,
    assignment: SdFmTrainAssignment,
) -> Result<TrainedSdFm> {
    let n = y.nrows();
    let d = y.ncols();
    if n == 0 || d == 0 {
        return Err(Error::Domain("y must be non-empty"));
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
    if !(fm_cfg.lr > 0.0) || !fm_cfg.lr.is_finite() {
        return Err(Error::Domain("lr must be positive and finite"));
    }
    if fm_cfg.steps == 0 || fm_cfg.batch_size == 0 {
        return Err(Error::Domain("steps and batch_size must be >= 1"));
    }

    // Normalize b for downstream uses.
    let b_norm = b.to_owned() / bs;

    // 1) Fit potentials (only needed for semidiscrete assignment).
    let g = match assignment {
        SdFmTrainAssignment::SemidiscretePotentials => {
            fit_potentials_sgd_neg_dot(y, &b_norm.view(), pot_cfg).map_err(|_| {
                Error::Domain("failed to fit semidiscrete potentials (see wass::semidiscrete)")
            })?
        }
        SdFmTrainAssignment::CategoricalFromB => Array1::<f32>::zeros(n),
    };

    // 2) Train conditional field with flow matching regression.
    let mut field = LinearCondField::new_zeros(d);

    let mut rng = ChaCha8Rng::seed_from_u64(fm_cfg.seed);

    for _step in 0..fm_cfg.steps {
        for _ in 0..fm_cfg.batch_size {
            // sample x0 ~ N(0,I)
            let mut x0 = Array1::<f32>::zeros(d);
            for k in 0..d {
                x0[k] = StandardNormal.sample(&mut rng);
            }

            // pick target y_j
            let j = match assignment {
                SdFmTrainAssignment::SemidiscretePotentials => {
                    let scores = scores_neg_dot(&x0.view(), y, &g.view());
                    assign_hard_from_scores(&scores.view())
                }
                SdFmTrainAssignment::CategoricalFromB => {
                    // Sample j ~ b_norm (explicitly normalized above).
                    sample_categorical_from_probs(&b_norm.view(), &mut rng)
                }
            };
            let yj = y.row(j);

            // random t in [0,1] (schedule matters for few-step behavior)
            let t: f32 = fm_cfg.t_schedule.sample_t(&mut rng);

            // linear interpolation path x_t = (1-t)x0 + t y
            let mut xt = Array1::<f32>::zeros(d);
            for k in 0..d {
                xt[k] = (1.0 - t) * x0[k] + t * yj[k];
            }

            // flow matching target for linear path: u = y - x0
            let mut u = Array1::<f32>::zeros(d);
            for k in 0..d {
                u[k] = yj[k] - x0[k];
            }

            field.sgd_step(&xt.view(), t, &yj, &u.view(), fm_cfg.lr);
        }
    }

    Ok(TrainedSdFm {
        y: y.to_owned(),
        b: b_norm,
        g,
        assignment,
        field,
    })
}

/// Train a **rectified flow matching** (RFM) baseline using **minibatch OT pairing**.
///
/// High-level idea (matches the RFM framing in arXiv:2507.17731 / the curated taxonomy):
/// - sample a minibatch of base noise points `x0_i ~ N(0, I)`
/// - sample a minibatch of target points `y_{j_i}` according to weights `b`
/// - compute an OT coupling between the two minibatches, then extract a one-to-one pairing
/// - train on straight-line paths between paired points
///
/// This does **not** try to be a full reproduction; it’s a minimal, testable primitive.
pub fn train_rfm_minibatch_ot_linear(
    y: &ArrayView2<f32>,
    b: &ArrayView1<f32>,
    rfm_cfg: &RfmMinibatchOtConfig,
    fm_cfg: &SdFmTrainConfig,
) -> Result<TrainedSdFm> {
    let n = y.nrows();
    let d = y.ncols();
    if n == 0 || d == 0 {
        return Err(Error::Domain("y must be non-empty"));
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
    if !(fm_cfg.lr > 0.0) || !fm_cfg.lr.is_finite() {
        return Err(Error::Domain("lr must be positive and finite"));
    }
    if fm_cfg.steps == 0 || fm_cfg.batch_size == 0 {
        return Err(Error::Domain("steps and batch_size must be >= 1"));
    }
    if rfm_cfg.pairing_every == 0 {
        return Err(Error::Domain("rfm_cfg.pairing_every must be >= 1"));
    }
    match rfm_cfg.pairing {
        RfmMinibatchPairing::SinkhornGreedy => {
            if !(rfm_cfg.reg > 0.0) || !rfm_cfg.reg.is_finite() {
                return Err(Error::Domain("rfm_cfg.reg must be positive and finite"));
            }
            if rfm_cfg.max_iter == 0 {
                return Err(Error::Domain("rfm_cfg.max_iter must be >= 1"));
            }
            if !(rfm_cfg.tol > 0.0) || !rfm_cfg.tol.is_finite() {
                return Err(Error::Domain("rfm_cfg.tol must be positive and finite"));
            }
        }
        RfmMinibatchPairing::SinkhornSelective { keep_frac } => {
            if !(rfm_cfg.reg > 0.0) || !rfm_cfg.reg.is_finite() {
                return Err(Error::Domain("rfm_cfg.reg must be positive and finite"));
            }
            if rfm_cfg.max_iter == 0 {
                return Err(Error::Domain("rfm_cfg.max_iter must be >= 1"));
            }
            if !(rfm_cfg.tol > 0.0) || !rfm_cfg.tol.is_finite() {
                return Err(Error::Domain("rfm_cfg.tol must be positive and finite"));
            }
            if !(keep_frac > 0.0) || !keep_frac.is_finite() {
                return Err(Error::Domain(
                    "rfm_cfg.keep_frac must be positive and finite",
                ));
            }
        }
        RfmMinibatchPairing::RowwiseNearest => {}
        RfmMinibatchPairing::ExpGreedy { temp } => {
            if !(temp > 0.0) || !temp.is_finite() {
                return Err(Error::Domain("rfm_cfg.temp must be positive and finite"));
            }
        }
        RfmMinibatchPairing::PartialRowwise { keep_frac } => {
            if !(keep_frac > 0.0) || !keep_frac.is_finite() {
                return Err(Error::Domain(
                    "rfm_cfg.keep_frac must be positive and finite",
                ));
            }
        }
    }

    let b_norm = b.to_owned() / bs;

    // No semidiscrete potentials in this baseline.
    let g = Array1::<f32>::zeros(n);

    let mut field = LinearCondField::new_zeros(d);
    let mut rng = ChaCha8Rng::seed_from_u64(fm_cfg.seed);

    // Pre-alloc minibatch buffers.
    let bs = fm_cfg.batch_size;
    let mut x0s = Array2::<f32>::zeros((bs, d));
    let mut ys = Array2::<f32>::zeros((bs, d));
    let mut js = vec![0usize; bs];

    let mut perm: Vec<usize> = Vec::new();
    for step in 0..fm_cfg.steps {
        // If we are amortizing coupling, we must also amortize the sampled minibatch.
        // Otherwise we'd be pairing *different* x0s/ys with a stale permutation.
        let recompute = step == 0 || (step % rfm_cfg.pairing_every == 0);
        if recompute {
            // 1) Sample x0 ~ N(0, I).
            for i in 0..bs {
                for k in 0..d {
                    x0s[[i, k]] = StandardNormal.sample(&mut rng);
                }
            }

            // 2) Sample target indices j ~ b, build y-batch.
            for i in 0..bs {
                let j = sample_categorical_from_probs(&b_norm.view(), &mut rng);
                js[i] = j;
                let yj = y.row(j);
                for k in 0..d {
                    ys[[i, k]] = yj[k];
                }
            }

            // 3) Pair x0s <-> ys via selected minibatch coupling.
            perm = match rfm_cfg.pairing {
                RfmMinibatchPairing::SinkhornGreedy => minibatch_ot_greedy_pairing(
                    &x0s.view(),
                    &ys.view(),
                    rfm_cfg.reg,
                    rfm_cfg.max_iter,
                    rfm_cfg.tol,
                )?,
                RfmMinibatchPairing::SinkhornSelective { keep_frac } => {
                    minibatch_ot_selective_pairing(
                        &x0s.view(),
                        &ys.view(),
                        rfm_cfg.reg,
                        rfm_cfg.max_iter,
                        rfm_cfg.tol,
                        keep_frac,
                    )?
                }
                RfmMinibatchPairing::RowwiseNearest => {
                    minibatch_rowwise_nearest_pairing(&x0s.view(), &ys.view())?
                }
                RfmMinibatchPairing::ExpGreedy { temp } => {
                    minibatch_exp_greedy_pairing(&x0s.view(), &ys.view(), temp)?
                }
                RfmMinibatchPairing::PartialRowwise { keep_frac } => {
                    minibatch_partial_rowwise_pairing(&x0s.view(), &ys.view(), keep_frac)?
                }
            };
        }

        // 4) FM regression updates along straight line between paired points.
        for i in 0..bs {
            let p = perm[i];
            let x0 = x0s.row(i);
            let y1 = ys.row(p);

            // random t in [0,1] (schedule matters for few-step behavior)
            let t: f32 = fm_cfg.t_schedule.sample_t(&mut rng);

            // xt = (1-t)x0 + t y1
            let mut xt = Array1::<f32>::zeros(d);
            for k in 0..d {
                xt[k] = (1.0 - t) * x0[k] + t * y1[k];
            }

            // u = y1 - x0
            let mut u = Array1::<f32>::zeros(d);
            for k in 0..d {
                u[k] = y1[k] - x0[k];
            }

            // Condition on the paired y1.
            field.sgd_step(&xt.view(), t, &y1, &u.view(), fm_cfg.lr);
        }
    }

    Ok(TrainedSdFm {
        y: y.to_owned(),
        b: b_norm,
        g,
        assignment: SdFmTrainAssignment::CategoricalFromB,
        field,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn hash_f32_bits(xs: &[f32]) -> u64 {
        // Stable, cheap hash over f32 bit patterns (no floating rounding).
        // FNV-1a-ish.
        let mut h: u64 = 1469598103934665603;
        for &x in xs {
            h ^= x.to_bits() as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h
    }

    fn batch_signature(x0s: &Array2<f32>, ys: &Array2<f32>) -> u64 {
        let hx = hash_f32_bits(x0s.as_slice().unwrap_or(&[]));
        let hy = hash_f32_bits(ys.as_slice().unwrap_or(&[]));
        hx ^ hy.rotate_left(17)
    }

    /// Trace the minibatch sampling schedule for RFM, ignoring coupling + SGD.
    ///
    /// We use this to assert the *intended* semantics of `pairing_every`:
    /// if we amortize coupling for k steps, we must also amortize the minibatch for k steps.
    fn trace_rfm_minibatch_schedule(
        y: &ArrayView2<f32>,
        b: &ArrayView1<f32>,
        rfm_cfg: &RfmMinibatchOtConfig,
        fm_cfg: &SdFmTrainConfig,
    ) -> Result<Vec<u64>> {
        let n = y.nrows();
        let d = y.ncols();
        if n == 0 || d == 0 {
            return Err(Error::Domain("y must be non-empty"));
        }
        if b.len() != n {
            return Err(Error::Shape("b length must match y.nrows()"));
        }
        let bs = b.sum();
        if bs <= 0.0 {
            return Err(Error::Domain("b must have positive total mass"));
        }
        let b_norm = b.to_owned() / bs;

        let bs = fm_cfg.batch_size;
        let mut rng = ChaCha8Rng::seed_from_u64(fm_cfg.seed);

        let mut x0s = Array2::<f32>::zeros((bs, d));
        let mut ys = Array2::<f32>::zeros((bs, d));
        let mut js = vec![0usize; bs];

        let mut sigs: Vec<u64> = Vec::with_capacity(fm_cfg.steps);

        for step in 0..fm_cfg.steps {
            let recompute = step == 0 || (step % rfm_cfg.pairing_every == 0);
            if recompute {
                for i in 0..bs {
                    for k in 0..d {
                        x0s[[i, k]] = StandardNormal.sample(&mut rng);
                    }
                }
                for i in 0..bs {
                    let j = sample_categorical_from_probs(&b_norm.view(), &mut rng);
                    js[i] = j;
                    let yj = y.row(j);
                    for k in 0..d {
                        ys[[i, k]] = yj[k];
                    }
                }
                // Strong internal-consistency check: js and ys must agree exactly.
                // (If this ever fails, training is conditioning on a different y than it thinks.)
                for i in 0..bs {
                    let j = js[i];
                    let yj = y.row(j);
                    for k in 0..d {
                        if ys[[i, k]].to_bits() != yj[k].to_bits() {
                            return Err(Error::Domain("inconsistent minibatch: ys[i] != y[js[i]]"));
                        }
                    }
                }
            } else {
                // Do nothing: x0s/ys must remain identical.
                let _ = &js;
            }
            sigs.push(batch_signature(&x0s, &ys));
        }

        Ok(sigs)
    }

    proptest! {
        #[test]
        fn prop_sample_categorical_in_range_and_deterministic(
            n in 1usize..128,
            seed in any::<u64>(),
        ) {
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;

            // Simple normalized distribution: p[i] ∝ 1/(i+1).
            let mut p = Array1::<f32>::zeros(n);
            for i in 0..n {
                p[i] = 1.0 / ((i + 1) as f32);
            }
            let s = p.sum();
            p = p / s;

            let mut r1 = ChaCha8Rng::seed_from_u64(seed);
            let mut r2 = ChaCha8Rng::seed_from_u64(seed);
            for _ in 0..256 {
                let a = sample_categorical_from_probs(&p.view(), &mut r1);
                let b = sample_categorical_from_probs(&p.view(), &mut r2);
                prop_assert!(a < n);
                prop_assert!(b < n);
                prop_assert_eq!(a, b);
            }
        }
    }

    proptest! {
        // This is the exact seam where we were previously wrong:
        // when pairing is amortized, the minibatch must be amortized too.
        #[test]
        fn prop_pairing_every_implies_batch_reuse(
            n in 10usize..40,
            d in 1usize..8,
            steps in 4usize..40,
            batch_size in 4usize..32,
            pairing_every in 1usize..8,
            seed in any::<u64>(),
        ) {
            let mut y = Array2::<f32>::zeros((n, d));
            // Deterministic pseudo-random-ish y so this stays fast and stable.
            for i in 0..n {
                for k in 0..d {
                    y[[i, k]] = (((i * 53 + k * 19) % 101) as f32 / 101.0) * 2.0 - 1.0;
                }
            }
            let b = Array1::<f32>::from_elem(n, 1.0);

            let fm_cfg = SdFmTrainConfig {
                lr: 1e-2,
                steps,
                batch_size,
                sample_steps: 10,
                seed,
                t_schedule: TimestepSchedule::Uniform,
            };
            let rfm_cfg = RfmMinibatchOtConfig {
                reg: 0.2,
                max_iter: 200,
                tol: 2e-2,
                pairing: RfmMinibatchPairing::RowwiseNearest,
                pairing_every,
            };

            let sigs = trace_rfm_minibatch_schedule(&y.view(), &b.view(), &rfm_cfg, &fm_cfg).unwrap();
            prop_assert_eq!(sigs.len(), steps);

            // Within each pairing_every-sized block, signatures must be identical.
            for t in 1..steps {
                if t % pairing_every != 0 {
                    prop_assert_eq!(
                        sigs[t],
                        sigs[t - 1],
                        "expected batch reuse at t={} pairing_every={}",
                        t,
                        pairing_every
                    );
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 12,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_pairing_every_implies_batch_reuse_exp_greedy(
            n in 10usize..30,
            d in 1usize..6,
            steps in 4usize..24,
            batch_size in 4usize..16,
            pairing_every in 1usize..8,
            seed in any::<u64>(),
        ) {
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    y[[i, k]] = (((i * 53 + k * 19) % 101) as f32 / 101.0) * 2.0 - 1.0;
                }
            }
            let b = Array1::<f32>::from_elem(n, 1.0);

            let fm_cfg = SdFmTrainConfig {
                lr: 1e-2,
                steps,
                batch_size,
                sample_steps: 10,
                seed,
                t_schedule: TimestepSchedule::Uniform,
            };
            let rfm_cfg = RfmMinibatchOtConfig {
                reg: 0.2,
                max_iter: 1_000,
                tol: 2e-2,
                pairing: RfmMinibatchPairing::ExpGreedy { temp: 0.2 },
                pairing_every,
            };

            let sigs = trace_rfm_minibatch_schedule(&y.view(), &b.view(), &rfm_cfg, &fm_cfg).unwrap();
            prop_assert_eq!(sigs.len(), steps);
            for t in 1..steps {
                if t % pairing_every != 0 {
                    prop_assert_eq!(sigs[t], sigs[t - 1], "expected batch reuse at t={} pairing_every={}", t, pairing_every);
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 10,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_pairing_every_implies_batch_reuse_sinkhorn_greedy(
            n in 10usize..24,
            d in 1usize..5,
            steps in 4usize..18,
            batch_size in 4usize..12,
            pairing_every in 1usize..6,
            seed in any::<u64>(),
        ) {
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    y[[i, k]] = (((i * 53 + k * 19) % 101) as f32 / 101.0) * 2.0 - 1.0;
                }
            }
            let b = Array1::<f32>::from_elem(n, 1.0);

            let fm_cfg = SdFmTrainConfig {
                lr: 1e-2,
                steps,
                batch_size,
                sample_steps: 10,
                seed,
                t_schedule: TimestepSchedule::Uniform,
            };
            let rfm_cfg = RfmMinibatchOtConfig {
                reg: 0.2,
                max_iter: 5_000,
                tol: 2e-2,
                pairing: RfmMinibatchPairing::SinkhornGreedy,
                pairing_every,
            };

            let sigs = trace_rfm_minibatch_schedule(&y.view(), &b.view(), &rfm_cfg, &fm_cfg).unwrap();
            prop_assert_eq!(sigs.len(), steps);
            for t in 1..steps {
                if t % pairing_every != 0 {
                    prop_assert_eq!(sigs[t], sigs[t - 1], "expected batch reuse at t={} pairing_every={}", t, pairing_every);
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 10,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_pairing_every_implies_batch_reuse_partial_rowwise(
            n in 10usize..30,
            d in 1usize..6,
            steps in 4usize..24,
            batch_size in 4usize..16,
            pairing_every in 1usize..8,
            seed in any::<u64>(),
        ) {
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    y[[i, k]] = (((i * 53 + k * 19) % 101) as f32 / 101.0) * 2.0 - 1.0;
                }
            }
            let b = Array1::<f32>::from_elem(n, 1.0);

            let fm_cfg = SdFmTrainConfig {
                lr: 1e-2,
                steps,
                batch_size,
                sample_steps: 10,
                seed,
                t_schedule: TimestepSchedule::Uniform,
            };
            let rfm_cfg = RfmMinibatchOtConfig {
                reg: 0.2,
                max_iter: 1_000,
                tol: 2e-2,
                pairing: RfmMinibatchPairing::PartialRowwise { keep_frac: 0.8 },
                pairing_every,
            };

            let sigs = trace_rfm_minibatch_schedule(&y.view(), &b.view(), &rfm_cfg, &fm_cfg).unwrap();
            prop_assert_eq!(sigs.len(), steps);
            for t in 1..steps {
                if t % pairing_every != 0 {
                    prop_assert_eq!(sigs[t], sigs[t - 1], "expected batch reuse at t={} pairing_every={}", t, pairing_every);
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 8,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_pairing_every_implies_batch_reuse_sinkhorn_selective(
            n in 10usize..22,
            d in 1usize..5,
            steps in 4usize..18,
            batch_size in 4usize..12,
            pairing_every in 1usize..6,
            seed in any::<u64>(),
        ) {
            let mut y = Array2::<f32>::zeros((n, d));
            for i in 0..n {
                for k in 0..d {
                    y[[i, k]] = (((i * 53 + k * 19) % 101) as f32 / 101.0) * 2.0 - 1.0;
                }
            }
            let b = Array1::<f32>::from_elem(n, 1.0);

            let fm_cfg = SdFmTrainConfig {
                lr: 1e-2,
                steps,
                batch_size,
                sample_steps: 10,
                seed,
                t_schedule: TimestepSchedule::Uniform,
            };
            let rfm_cfg = RfmMinibatchOtConfig {
                reg: 0.2,
                max_iter: 5_000,
                tol: 2e-2,
                pairing: RfmMinibatchPairing::SinkhornSelective { keep_frac: 0.8 },
                pairing_every,
            };

            let sigs = trace_rfm_minibatch_schedule(&y.view(), &b.view(), &rfm_cfg, &fm_cfg).unwrap();
            prop_assert_eq!(sigs.len(), steps);
            for t in 1..steps {
                if t % pairing_every != 0 {
                    prop_assert_eq!(sigs[t], sigs[t - 1], "expected batch reuse at t={} pairing_every={}", t, pairing_every);
                }
            }
        }
    }

    #[test]
    fn ushaped_has_more_mass_near_boundaries_than_uniform() {
        // Paper nuance: U-shaped schedules emphasize boundary timesteps.
        // This is a cheap regression guard on the distributional shape.
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let n = 50_000usize;
        let eps = 0.05f32;

        let mut near_u = 0usize;
        let mut near_uni = 0usize;

        for _ in 0..n {
            let tu = TimestepSchedule::UShaped.sample_t(&mut rng);
            let t0 = TimestepSchedule::Uniform.sample_t(&mut rng);
            if tu <= eps || tu >= 1.0 - eps {
                near_u += 1;
            }
            if t0 <= eps || t0 >= 1.0 - eps {
                near_uni += 1;
            }
        }

        let fu = near_u as f32 / n as f32;
        let f0 = near_uni as f32 / n as f32;

        // Uniform has ~2*eps mass in the boundary region; U-shaped should be noticeably higher.
        assert!(
            fu > f0 + 0.10,
            "expected UShaped to concentrate near boundaries: fu={fu:.3} f0={f0:.3}"
        );
    }
}
