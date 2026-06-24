#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("This example requires the `burn` feature.");
    eprintln!(
        "Run: cargo run -p flowmatch --example burn_sd_fm_semidiscrete_linear --features burn"
    );
}

#[cfg(feature = "burn")]
fn main() {
    use burn_core::tensor::backend::Backend;
    use flowmatch::burn_sd_fm::{train_sd_fm_semidiscrete_linear_burn, BurnBackend};
    use flowmatch::sd_fm::{SdFmTrainAssignment, SdFmTrainConfig, TimestepSchedule};
    use ndarray::{Array1, Array2};
    use wass::semidiscrete::SemidiscreteSgdConfig;

    let device = <BurnBackend as Backend>::Device::default();

    // Tiny toy support in R^2 (four corners), uniform weights.
    let y = Array2::<f32>::from_shape_vec(
        (4, 2),
        vec![
            -1.0, -1.0, //
            1.0, -1.0, //
            -1.0, 1.0, //
            1.0, 1.0, //
        ],
    )
    .unwrap();
    let b = Array1::<f32>::from_elem(4, 1.0);

    let pot_cfg = SemidiscreteSgdConfig::default();
    let fm_cfg = SdFmTrainConfig {
        lr: 5e-3,
        steps: 50,
        batch_size: 64,
        sample_steps: 0,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };

    let trained = train_sd_fm_semidiscrete_linear_burn(
        &device,
        &y.view(),
        &b.view(),
        &pot_cfg,
        &fm_cfg,
        SdFmTrainAssignment::CategoricalFromB,
        1e-2,
    )
    .unwrap();

    println!(
        "trained burn SD-FM (exported to ndarray): field W shape = {:?}",
        trained.field.w().dim()
    );

    // Proof the export trained a useful field: integrating x0 ~ N(0, I) through
    // the learned field must land samples closer to their assigned target than
    // the raw x0 noise was. A field shape alone proves nothing -- only the
    // before/after distance does.
    let n_samp = 64usize;
    let ode_steps = 30usize;
    let (x0s, x1s, js) = trained.sample_with_x0(n_samp, 123, ode_steps).unwrap();
    let mse = |xs: &ndarray::Array2<f32>| -> f64 {
        let (n, d) = (xs.nrows(), xs.ncols());
        let mut s = 0.0f64;
        for i in 0..n {
            let j = js[i];
            for k in 0..d {
                let r = (xs[[i, k]] - trained.y[[j, k]]) as f64;
                s += r * r;
            }
        }
        s / (n as f64 * d as f64)
    };
    let baseline_mse = mse(&x0s);
    let trained_mse = mse(&x1s);
    println!("  baseline (x0)  mse to assigned y = {baseline_mse:.4}");
    println!("  trained (x1)   mse to assigned y = {trained_mse:.4}");
    assert!(
        trained_mse < baseline_mse,
        "burn SD-FM training did not improve over the x0 baseline: \
         trained={trained_mse:.4} >= baseline={baseline_mse:.4}"
    );
}
