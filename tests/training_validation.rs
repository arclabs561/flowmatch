use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, train_sd_fm_semidiscrete_linear_with_assignment,
    RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainAssignment, SdFmTrainConfig,
    TimestepSchedule,
};
use ndarray::{array, Array1, Array2};
use wass::semidiscrete::SemidiscreteSgdConfig;

fn valid_y() -> Array2<f32> {
    array![[0.0, 1.0], [1.0, 0.0]]
}

fn valid_b() -> Array1<f32> {
    array![1.0, 1.0]
}

fn valid_fm_config() -> SdFmTrainConfig {
    SdFmTrainConfig {
        lr: 1e-2,
        steps: 1,
        batch_size: 1,
        sample_steps: 1,
        seed: 7,
        t_schedule: TimestepSchedule::Uniform,
    }
}

fn assert_ndarray_trainers_reject(y: &Array2<f32>, b: &Array1<f32>, fm_cfg: &SdFmTrainConfig) {
    let pot_cfg = SemidiscreteSgdConfig::default();
    assert!(train_sd_fm_semidiscrete_linear_with_assignment(
        &y.view(),
        &b.view(),
        &pot_cfg,
        fm_cfg,
        SdFmTrainAssignment::CategoricalFromB,
    )
    .is_err());

    let rfm_cfg = RfmMinibatchOtConfig {
        pairing: RfmMinibatchPairing::RowwiseNearest,
        ..Default::default()
    };
    assert!(train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, fm_cfg).is_err());
}

#[test]
fn ndarray_trainers_reject_nonfinite_support_and_weights() {
    let y = valid_y();
    let b = valid_b();
    let fm_cfg = valid_fm_config();

    let y_nan = array![[f32::NAN, 1.0], [1.0, 0.0]];
    assert_ndarray_trainers_reject(&y_nan, &b, &fm_cfg);

    let y_inf = array![[f32::INFINITY, 1.0], [1.0, 0.0]];
    assert_ndarray_trainers_reject(&y_inf, &b, &fm_cfg);

    let b_nan = array![f32::NAN, 1.0];
    assert_ndarray_trainers_reject(&y, &b_nan, &fm_cfg);

    let b_inf = array![f32::INFINITY, 1.0];
    assert_ndarray_trainers_reject(&y, &b_inf, &fm_cfg);

    let b_overflow = array![f32::MAX, f32::MAX];
    assert_ndarray_trainers_reject(&y, &b_overflow, &fm_cfg);
}

#[test]
fn ndarray_trainers_reject_invalid_effective_learning_rate_and_schedule() {
    let y = valid_y();
    let b = valid_b();

    let mut invalid_lr = valid_fm_config();
    invalid_lr.lr = f32::NAN;
    assert_ndarray_trainers_reject(&y, &b, &invalid_lr);

    let mut zero_steps = valid_fm_config();
    zero_steps.steps = 0;
    assert_ndarray_trainers_reject(&y, &b, &zero_steps);

    let mut zero_batch = valid_fm_config();
    zero_batch.batch_size = 0;
    assert_ndarray_trainers_reject(&y, &b, &zero_batch);

    for schedule in [
        TimestepSchedule::LogitNormal {
            mean: f32::NAN,
            std: 1.0,
        },
        TimestepSchedule::LogitNormal {
            mean: 0.0,
            std: f32::NAN,
        },
        TimestepSchedule::LogitNormal {
            mean: 0.0,
            std: -1.0,
        },
    ] {
        let mut fm_cfg = valid_fm_config();
        fm_cfg.t_schedule = schedule;
        assert_ndarray_trainers_reject(&y, &b, &fm_cfg);
    }
}

#[cfg(feature = "burn")]
mod burn {
    use super::*;
    use burn_core::tensor::backend::Backend;
    use flowmatch::burn_sd_fm::{
        train_rfm_minibatch_ot_linear_burn, train_sd_fm_semidiscrete_linear_burn, BurnBackend,
    };

    fn assert_burn_trainers_reject(
        y: &Array2<f32>,
        b: &Array1<f32>,
        fm_cfg: &SdFmTrainConfig,
        lr: f64,
    ) {
        let device = <BurnBackend as Backend>::Device::default();
        let pot_cfg = SemidiscreteSgdConfig::default();
        assert!(train_sd_fm_semidiscrete_linear_burn(
            &device,
            &y.view(),
            &b.view(),
            &pot_cfg,
            fm_cfg,
            SdFmTrainAssignment::CategoricalFromB,
            lr,
        )
        .is_err());

        let rfm_cfg = RfmMinibatchOtConfig {
            pairing: RfmMinibatchPairing::RowwiseNearest,
            ..Default::default()
        };
        assert!(train_rfm_minibatch_ot_linear_burn(
            &device,
            &y.view(),
            &b.view(),
            &rfm_cfg,
            fm_cfg,
            lr,
        )
        .is_err());
    }

    #[test]
    fn burn_trainers_use_shared_input_validation_and_burn_learning_rate() {
        let y = valid_y();
        let b = valid_b();
        let fm_cfg = valid_fm_config();

        let y_nan = array![[f32::NAN, 1.0], [1.0, 0.0]];
        assert_burn_trainers_reject(&y_nan, &b, &fm_cfg, 1e-2);

        let b_overflow = array![f32::MAX, f32::MAX];
        assert_burn_trainers_reject(&y, &b_overflow, &fm_cfg, 1e-2);

        let mut invalid_schedule = valid_fm_config();
        invalid_schedule.t_schedule = TimestepSchedule::LogitNormal {
            mean: 0.0,
            std: f32::NAN,
        };
        assert_burn_trainers_reject(&y, &b, &invalid_schedule, 1e-2);

        assert_burn_trainers_reject(&y, &b, &fm_cfg, f64::NAN);
    }

    #[test]
    fn burn_trainers_ignore_unused_ndarray_learning_rate() {
        let device = <BurnBackend as Backend>::Device::default();
        let y = valid_y();
        let b = valid_b();
        let mut fm_cfg = valid_fm_config();
        fm_cfg.lr = f32::NAN;
        let pot_cfg = SemidiscreteSgdConfig::default();

        let sd = train_sd_fm_semidiscrete_linear_burn(
            &device,
            &y.view(),
            &b.view(),
            &pot_cfg,
            &fm_cfg,
            SdFmTrainAssignment::CategoricalFromB,
            1e-2,
        );
        assert!(
            sd.is_ok(),
            "Burn training must use its explicit learning rate"
        );

        let rfm_cfg = RfmMinibatchOtConfig {
            pairing: RfmMinibatchPairing::RowwiseNearest,
            ..Default::default()
        };
        let rfm = train_rfm_minibatch_ot_linear_burn(
            &device,
            &y.view(),
            &b.view(),
            &rfm_cfg,
            &fm_cfg,
            1e-2,
        );
        assert!(rfm.is_ok(), "Burn RFM must use its explicit learning rate");
    }
}

#[cfg(feature = "riemannian")]
mod riemannian {
    use super::*;
    use flowmatch::riemannian::{train_riemannian_fm, ManifoldVectorField};
    use hyperball::PoincareBall;
    use ndarray::{array, ArrayView1};

    #[derive(Default)]
    struct RecordingField {
        updates: usize,
    }

    impl ManifoldVectorField for RecordingField {
        fn eval(&self, x: &ArrayView1<f32>, _t: f32) -> Array1<f32> {
            Array1::zeros(x.len())
        }

        fn sgd_step(&mut self, _x: &ArrayView1<f32>, _t: f32, _target: &ArrayView1<f32>, _lr: f32) {
            self.updates += 1;
        }
    }

    fn manifold() -> PoincareBall<f64> {
        PoincareBall::new(1.0)
    }

    fn valid_targets() -> Array2<f32> {
        array![[0.1, 0.2]]
    }

    #[test]
    fn riemannian_rejects_invalid_config_before_sgd() {
        let targets = valid_targets();
        let mut cfg = valid_fm_config();
        cfg.t_schedule = TimestepSchedule::LogitNormal {
            mean: 0.0,
            std: f32::NAN,
        };
        let mut field = RecordingField::default();

        let result = train_riemannian_fm(&manifold(), &mut field, &targets.view(), &cfg, |_| {
            array![0.0, 0.0]
        });

        assert!(result.is_err());
        assert_eq!(field.updates, 0);
    }

    #[test]
    fn riemannian_rejects_invalid_base_samples_before_sgd() {
        let targets = valid_targets();
        let cfg = valid_fm_config();

        let mut wrong_width_field = RecordingField::default();
        let wrong_width = train_riemannian_fm(
            &manifold(),
            &mut wrong_width_field,
            &targets.view(),
            &cfg,
            |_| array![0.0],
        );
        assert!(wrong_width.is_err());
        assert_eq!(wrong_width_field.updates, 0);

        let mut nonfinite_field = RecordingField::default();
        let nonfinite = train_riemannian_fm(
            &manifold(),
            &mut nonfinite_field,
            &targets.view(),
            &cfg,
            |_| array![f32::NAN, 0.0],
        );
        assert!(nonfinite.is_err());
        assert_eq!(nonfinite_field.updates, 0);
    }

    #[test]
    fn riemannian_rejects_nonfinite_targets_before_sgd() {
        let targets = array![[f32::INFINITY, 0.2]];
        let cfg = valid_fm_config();
        let mut field = RecordingField::default();

        let result = train_riemannian_fm(&manifold(), &mut field, &targets.view(), &cfg, |_| {
            array![0.0, 0.0]
        });

        assert!(result.is_err());
        assert_eq!(field.updates, 0);
    }
}
