//! Burn-backed (opt-in) training loops for SD-FM / RFM that export to the existing ndarray model.
//!
//! Design goal: use Burn autodiff + optimizers to train, but keep the public model type
//! (`sd_fm::TrainedSdFm`) unchanged by exporting learned parameters back into
//! `linear::LinearCondField`.
//!
//! This makes Burn a backend for **training**, without forcing Burn tensor types into the default
//! API surface.

use burn_core as burn;

use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use burn_nn::{Linear, LinearConfig};
use burn_optim::{GradientsParams, LearningRate, Optimizer, SgdConfig};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use wass::semidiscrete::{
    assign_hard_from_scores, fit_potentials_sgd_neg_dot, scores_neg_dot, SemidiscreteSgdConfig,
};

use crate::linear::LinearCondField;
use crate::sd_fm::{SdFmTrainAssignment, SdFmTrainConfig, TrainedSdFm};
use crate::{Error, Result};

/// Default burn backend for these training loops: ndarray + autodiff.
pub type BurnBackend = Autodiff<NdArray<f32>>;

#[derive(Module, Debug)]
struct BurnLinearCondField<B: Backend> {
    linear: Linear<B>,
    d: usize,
}

impl<B: Backend> BurnLinearCondField<B> {
    fn new(device: &B::Device, d: usize) -> Self {
        // Features: [x_t (d), y (d), t (1)] => 2d + 1.
        let in_dim = 2 * d + 1;
        let linear = LinearConfig::new(in_dim, d).with_bias(true).init(device);
        Self { linear, d }
    }

    fn forward(&self, x_t: Tensor<B, 2>, y: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        let feats = Tensor::cat(vec![x_t, y, t], 1);
        self.linear.forward(feats)
    }

    fn export_to_ndarray(&self) -> LinearCondField {
        // Burn Linear weight is [d_input, d_output] = [(2d+1), d].
        // LinearCondField wants W as [d_output, (2d+2)] with a constant-bias feature.
        let w_data = self.linear.weight.to_data();
        let w_shape = &w_data.shape;
        debug_assert_eq!(w_shape.len(), 2);
        let d_in = w_shape[0];
        let d_out = w_shape[1];
        debug_assert_eq!(d_out, self.d);
        debug_assert_eq!(d_in, 2 * self.d + 1);

        let b = self
            .linear
            .bias
            .as_ref()
            .map(|b| b.to_data().to_vec::<f32>().expect("bias to_vec"))
            .unwrap_or_else(|| vec![0.0; d_out]);

        let w_flat: Vec<f32> = w_data.to_vec::<f32>().expect("weight to_vec");
        // w_flat is row-major in burn tensor data.
        // Index: w_flat[i * d_out + j] where i in [0,d_in), j in [0,d_out).

        let mut w = Array2::<f32>::zeros((d_out, 2 * self.d + 2));
        for j in 0..d_out {
            // Copy transposed weights into the first (2d+1) columns.
            for i in 0..d_in {
                w[[j, i]] = w_flat[i * d_out + j];
            }
            // Put bias into the final column (constant 1 feature in LinearCondField).
            w[[j, 2 * self.d + 1]] = b[j];
        }

        LinearCondField { w }
    }
}

fn sample_categorical_from_probs(probs: &ArrayView1<f32>, rng: &mut impl rand::Rng) -> usize {
    debug_assert!(probs.len() > 0);
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

fn ndarray_to_burn_2<B: Backend>(device: &B::Device, x: &Array2<f32>) -> Tensor<B, 2> {
    let (n, d) = x.dim();
    let data = burn::tensor::TensorData::new(x.as_slice().unwrap_or(&[]).to_vec(), [n, d]);
    Tensor::from_data(data, device)
}

fn ndarray_to_burn_2_keepdim<B: Backend>(device: &B::Device, x: &Array1<f32>) -> Tensor<B, 2> {
    // Shape [batch, 1]
    let n = x.len();
    let data = burn::tensor::TensorData::new(x.as_slice().unwrap_or(&[]).to_vec(), [n, 1]);
    Tensor::from_data(data, device)
}

/// Burn-backed version of SD-FM training (exports to `TrainedSdFm`).
pub fn train_sd_fm_semidiscrete_linear_burn(
    device: &<BurnBackend as Backend>::Device,
    y: &ArrayView2<f32>,
    b: &ArrayView1<f32>,
    pot_cfg: &SemidiscreteSgdConfig,
    fm_cfg: &SdFmTrainConfig,
    assignment: SdFmTrainAssignment,
    lr: LearningRate,
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
    if fm_cfg.steps == 0 || fm_cfg.batch_size == 0 {
        return Err(Error::Domain("steps and batch_size must be >= 1"));
    }

    let b_norm = b.to_owned() / bs;
    let g = match assignment {
        SdFmTrainAssignment::SemidiscretePotentials => {
            fit_potentials_sgd_neg_dot(y, &b_norm.view(), pot_cfg)
                .map_err(|_| Error::Domain("failed to fit semidiscrete potentials"))?
        }
        SdFmTrainAssignment::CategoricalFromB => Array1::<f32>::zeros(n),
    };

    let mut model = BurnLinearCondField::<BurnBackend>::new(device, d);
    let mut optim = SgdConfig::new().init::<BurnBackend, BurnLinearCondField<BurnBackend>>();

    let mut rng = ChaCha8Rng::seed_from_u64(fm_cfg.seed);

    // Batch buffers.
    let bs = fm_cfg.batch_size;
    let mut x0s = Array2::<f32>::zeros((bs, d));
    let mut ys = Array2::<f32>::zeros((bs, d));
    let mut ts = Array1::<f32>::zeros(bs);
    let mut xts = Array2::<f32>::zeros((bs, d));
    let mut us = Array2::<f32>::zeros((bs, d));

    for _step in 0..fm_cfg.steps {
        // Sample x0, y, t and build xt, u in ndarray (for deterministic assignment logic).
        for i in 0..bs {
            for k in 0..d {
                x0s[[i, k]] = StandardNormal.sample(&mut rng);
            }

            let x0 = x0s.row(i);
            let j = match assignment {
                SdFmTrainAssignment::SemidiscretePotentials => {
                    let scores = scores_neg_dot(&x0, y, &g.view());
                    assign_hard_from_scores(&scores.view())
                }
                SdFmTrainAssignment::CategoricalFromB => {
                    sample_categorical_from_probs(&b_norm.view(), &mut rng)
                }
            };
            let yj = y.row(j);
            for k in 0..d {
                ys[[i, k]] = yj[k];
            }

            let t = fm_cfg.t_schedule.sample_t(&mut rng);
            ts[i] = t;

            for k in 0..d {
                let x0k = x0s[[i, k]];
                let yk = ys[[i, k]];
                xts[[i, k]] = (1.0 - t) * x0k + t * yk;
                us[[i, k]] = yk - x0k;
            }
        }

        // Burn step.
        let x_t = ndarray_to_burn_2::<BurnBackend>(device, &xts);
        let y_b = ndarray_to_burn_2::<BurnBackend>(device, &ys);
        let t_b = ndarray_to_burn_2_keepdim::<BurnBackend>(device, &ts);
        let u_b = ndarray_to_burn_2::<BurnBackend>(device, &us);

        let pred = model.forward(x_t, y_b, t_b);
        let loss = (pred - u_b).powf_scalar(2.0).mean();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }

    let field = model.export_to_ndarray();

    Ok(TrainedSdFm {
        y: y.to_owned(),
        b: b_norm,
        g,
        assignment,
        field,
    })
}

/// Burn-backed version of RFM minibatch OT training (exports to `TrainedSdFm`).
///
/// We keep the coupling logic (pairing) in ndarray, then perform the regression update in Burn.
pub fn train_rfm_minibatch_ot_linear_burn(
    device: &<BurnBackend as Backend>::Device,
    y: &ArrayView2<f32>,
    b: &ArrayView1<f32>,
    rfm_cfg: &crate::sd_fm::RfmMinibatchOtConfig,
    fm_cfg: &SdFmTrainConfig,
    lr: LearningRate,
) -> Result<TrainedSdFm> {
    // Reuse the validation and pairing logic from ndarray implementation, but do the SGD step in Burn.
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
    if fm_cfg.steps == 0 || fm_cfg.batch_size == 0 {
        return Err(Error::Domain("steps and batch_size must be >= 1"));
    }
    if rfm_cfg.pairing_every == 0 {
        return Err(Error::Domain("rfm_cfg.pairing_every must be >= 1"));
    }

    let b_norm = b.to_owned() / bs;
    let g = Array1::<f32>::zeros(n);

    let mut model = BurnLinearCondField::<BurnBackend>::new(device, d);
    let mut optim = SgdConfig::new().init::<BurnBackend, BurnLinearCondField<BurnBackend>>();
    let mut rng = ChaCha8Rng::seed_from_u64(fm_cfg.seed);

    let bs = fm_cfg.batch_size;
    let mut x0s = Array2::<f32>::zeros((bs, d));
    let mut ys = Array2::<f32>::zeros((bs, d));
    let mut perm: Vec<usize> = Vec::new();

    let mut ts = Array1::<f32>::zeros(bs);
    let mut xts = Array2::<f32>::zeros((bs, d));
    let mut us = Array2::<f32>::zeros((bs, d));

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
                let yj = y.row(j);
                for k in 0..d {
                    ys[[i, k]] = yj[k];
                }
            }

            perm = match rfm_cfg.pairing {
                crate::sd_fm::RfmMinibatchPairing::SinkhornGreedy => {
                    crate::rfm::minibatch_ot_greedy_pairing(
                        &x0s.view(),
                        &ys.view(),
                        rfm_cfg.reg,
                        rfm_cfg.max_iter,
                        rfm_cfg.tol,
                    )?
                }
                crate::sd_fm::RfmMinibatchPairing::SinkhornSelective { keep_frac } => {
                    crate::rfm::minibatch_ot_selective_pairing(
                        &x0s.view(),
                        &ys.view(),
                        rfm_cfg.reg,
                        rfm_cfg.max_iter,
                        rfm_cfg.tol,
                        keep_frac,
                    )?
                }
                crate::sd_fm::RfmMinibatchPairing::RowwiseNearest => {
                    crate::rfm::minibatch_rowwise_nearest_pairing(&x0s.view(), &ys.view())?
                }
                crate::sd_fm::RfmMinibatchPairing::ExpGreedy { temp } => {
                    crate::rfm::minibatch_exp_greedy_pairing(&x0s.view(), &ys.view(), temp)?
                }
                crate::sd_fm::RfmMinibatchPairing::PartialRowwise { keep_frac } => {
                    crate::rfm::minibatch_partial_rowwise_pairing(
                        &x0s.view(),
                        &ys.view(),
                        keep_frac,
                    )?
                }
            };
        }

        // Build the regression batch aligned with the cached permutation.
        for i in 0..bs {
            let p = perm[i];
            let t = fm_cfg.t_schedule.sample_t(&mut rng);
            ts[i] = t;
            for k in 0..d {
                let x0k = x0s[[i, k]];
                let yk = ys[[p, k]];
                xts[[i, k]] = (1.0 - t) * x0k + t * yk;
                us[[i, k]] = yk - x0k;
            }
        }

        let x_t = ndarray_to_burn_2::<BurnBackend>(device, &xts);
        let y1 = ndarray_to_burn_2::<BurnBackend>(device, &ys.select(ndarray::Axis(0), &perm));
        let t_b = ndarray_to_burn_2_keepdim::<BurnBackend>(device, &ts);
        let u_b = ndarray_to_burn_2::<BurnBackend>(device, &us);

        let pred = model.forward(x_t, y1, t_b);
        let loss = (pred - u_b).powf_scalar(2.0).mean();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }

    let field = model.export_to_ndarray();

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

    #[test]
    fn burn_sd_fm_trains_and_exports_linear_field() {
        let device = <BurnBackend as Backend>::Device::default();
        let y = Array2::<f32>::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
        let b = Array1::<f32>::from_elem(4, 1.0);
        let pot_cfg = SemidiscreteSgdConfig::default();
        let fm_cfg = SdFmTrainConfig {
            steps: 3,
            batch_size: 16,
            ..Default::default()
        };

        let m = train_sd_fm_semidiscrete_linear_burn(
            &device,
            &y.view(),
            &b.view(),
            &pot_cfg,
            &fm_cfg,
            SdFmTrainAssignment::CategoricalFromB,
            1e-2,
        )
        .unwrap();

        assert_eq!(m.field.w.nrows(), 2);
        assert_eq!(m.field.w.ncols(), 2 * 2 + 2);
    }
}
