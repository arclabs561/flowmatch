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
        trained.field.w.dim()
    );
}
