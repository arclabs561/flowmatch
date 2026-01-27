//! Burn-backed (opt-in) Euclidean flow matching utilities.
//!
//! This module is intentionally **additive**: it provides a concrete `burn` backend path
//! without changing the default `ndarray`-only API surface of `flowmatch`.
//!
//! Current scope:
//! - A tiny conditional vector field (`BurnEuclideanCondMlp`) over a straight-line path.
//! - Minimal helpers to compute \(x_t = (1-t)x_0 + t x_1\) and \(u_t = x_1 - x_0\).
//!
//! Near-term roadmap:
//! - Replace the “toy” training stub with a real Burn optimizer loop.
//! - Add a `burn`-backed Riemannian FM variant once manifold ops have a tensor backend.

use burn_core as burn;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use burn_nn::{Linear, LinearConfig, Relu};
use burn_optim::{GradientsParams, LearningRate, Optimizer, SgdConfig};

/// Default burn backend for this crate’s examples: ndarray + autodiff.
pub type BurnBackend = Autodiff<NdArray<f32>>;

/// A tiny conditional MLP vector field for Euclidean flow matching.
///
/// Input features are concatenated as: `[x_t, x_1, t, 1]` (so input dim is `2d + 2`).
/// Output is a vector field in `R^d`.
#[derive(Module, Debug)]
pub struct BurnEuclideanCondMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: Backend> BurnEuclideanCondMlp<B> {
    /// Initialize a small 2-layer MLP for dimension `d`.
    pub fn new(device: &B::Device, d: usize, hidden: usize) -> Self {
        let in_dim = 2 * d + 2;
        let l1 = LinearConfig::new(in_dim, hidden).init(device);
        let l2 = LinearConfig::new(hidden, d).init(device);
        Self { l1, l2 }
    }

    /// Forward pass for a batch.
    ///
    /// Shapes:
    /// - `x_t`: `[batch, d]`
    /// - `x1`: `[batch, d]`
    /// - `t`: `[batch, 1]` (column vector)
    /// Returns: `[batch, d]`.
    pub fn forward(&self, x_t: Tensor<B, 2>, x1: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        // TODO: Avoid repeated concatenation allocations (use fused ops / views if/when available).
        let ones = Tensor::<B, 2>::ones([t.dims()[0], 1], &t.device());
        let feats = Tensor::cat(vec![x_t, x1, t, ones], 1);
        let h = Relu.forward(self.l1.forward(feats));
        self.l2.forward(h)
    }
}

/// Straight-line path \(x_t = (1-t)x_0 + t x_1\) and target velocity \(u_t = x_1 - x_0\).
///
/// `t` is expected as `[batch, 1]`.
pub fn euclidean_path_targets<B: Backend>(
    x0: Tensor<B, 2>,
    x1: Tensor<B, 2>,
    t: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let xt =
        x0.clone() * (Tensor::ones([t.dims()[0], 1], &t.device()) - t.clone()) + x1.clone() * t;
    let ut = x1 - x0;
    (xt, ut)
}

/// Minimal SGD training loop (Burn autodiff backend).
///
/// This is meant as a compile-checked “migration foothold”: it proves we can express a full
/// FM regression step (targets, loss, autodiff backward, optimizer step) using Burn.
pub fn train_euclidean_fm_sgd(
    device: &<BurnBackend as Backend>::Device,
    d: usize,
    hidden: usize,
    steps: usize,
    batch_size: usize,
    lr: LearningRate,
) -> BurnEuclideanCondMlp<BurnBackend> {
    use burn::tensor::Distribution;

    let mut model = BurnEuclideanCondMlp::<BurnBackend>::new(device, d, hidden);
    let config = SgdConfig::new();
    let mut optim = config.init::<BurnBackend, BurnEuclideanCondMlp<BurnBackend>>();

    for _ in 0..steps {
        let x0 = Tensor::<BurnBackend, 2>::random(
            [batch_size, d],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let x1 = Tensor::<BurnBackend, 2>::random(
            [batch_size, d],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let t = Tensor::<BurnBackend, 2>::random(
            [batch_size, 1],
            Distribution::Uniform(0.0, 1.0),
            device,
        );

        let (xt, ut) = euclidean_path_targets(x0, x1.clone(), t.clone());
        let pred = model.forward(xt, x1, t);
        let loss = (pred - ut).powf_scalar(2.0).mean();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }

    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn burn_euclidean_shapes_smoke() {
        let device = <BurnBackend as Backend>::Device::default();

        let batch = 4usize;
        let d = 3usize;

        let x0 =
            Tensor::<BurnBackend, 2>::random([batch, d], Distribution::Normal(0.0, 1.0), &device);
        let x1 =
            Tensor::<BurnBackend, 2>::random([batch, d], Distribution::Normal(0.0, 1.0), &device);
        let t =
            Tensor::<BurnBackend, 2>::random([batch, 1], Distribution::Uniform(0.0, 1.0), &device);

        let (xt, ut) = euclidean_path_targets(x0, x1.clone(), t.clone());
        assert_eq!(xt.dims(), [batch, d]);
        assert_eq!(ut.dims(), [batch, d]);

        let model = BurnEuclideanCondMlp::<BurnBackend>::new(&device, d, 8);
        let pred = model.forward(xt, x1, t);
        assert_eq!(pred.dims(), [batch, d]);
    }

    #[test]
    fn burn_euclidean_train_smoke() {
        let device = <BurnBackend as Backend>::Device::default();
        let _model = train_euclidean_fm_sgd(&device, 4, 16, 2, 8, 1e-2);
    }
}
