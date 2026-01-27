#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("This example requires the `burn` feature.");
    eprintln!("Run: cargo run -p flowmatch --example burn_rfm_minibatch_ot_linear --features burn");
}

#[cfg(feature = "burn")]
fn main() {
    use burn_core::tensor::backend::Backend;
    use flowmatch::burn_sd_fm::{train_rfm_minibatch_ot_linear_burn, BurnBackend};
    use flowmatch::sd_fm::{
        RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig, TimestepSchedule,
    };
    use ndarray::{Array1, Array2};

    let device = <BurnBackend as Backend>::Device::default();

    // Tiny toy dataset in R^2: a jittered circle-ish set with uniform weights.
    let y = Array2::<f32>::from_shape_vec(
        (8, 2),
        vec![
            1.0, 0.0, 0.7, 0.7, 0.0, 1.0, -0.7, 0.7, -1.0, 0.0, -0.7, -0.7, 0.0, -1.0, 0.7, -0.7,
        ],
    )
    .unwrap();
    let b = Array1::<f32>::from_elem(8, 1.0);

    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 0.2,
        max_iter: 1000,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::RowwiseNearest,
        pairing_every: 1,
    };
    let fm_cfg = SdFmTrainConfig {
        lr: 5e-3,
        steps: 50,
        batch_size: 32,
        sample_steps: 0,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };

    let trained =
        train_rfm_minibatch_ot_linear_burn(&device, &y.view(), &b.view(), &rfm_cfg, &fm_cfg, 1e-2)
            .unwrap();

    println!(
        "trained burn RFM (exported to ndarray): field W shape = {:?}",
        trained.field.w.dim()
    );
}
