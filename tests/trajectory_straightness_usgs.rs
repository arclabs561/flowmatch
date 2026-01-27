use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, train_sd_fm_semidiscrete_linear_with_assignment,
    RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainAssignment, SdFmTrainConfig,
    TimestepSchedule,
};
use flowmatch::Result;
use ndarray::{Array1, Array2};
use wass::semidiscrete::SemidiscreteSgdConfig;

const USGS_CSV: &str = include_str!("../examples_data/usgs_eq_m6_2024_limit50.csv.txt");

fn deg_to_rad(x: f32) -> f32 {
    x * core::f32::consts::PI / 180.0
}

fn latlon_to_unit_xyz(lat_deg: f32, lon_deg: f32) -> [f32; 3] {
    let lat = deg_to_rad(lat_deg);
    let lon = deg_to_rad(lon_deg);
    let clat = lat.cos();
    [clat * lon.cos(), clat * lon.sin(), lat.sin()]
}

fn l2(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

fn integrate_euler_path(
    x0: &Array1<f32>,
    yj: &ndarray::ArrayView1<'_, f32>,
    field: &flowmatch::linear::LinearCondField,
    steps: usize,
) -> (f32, f32) {
    // Return (arc_length, chord_length)
    let mut x = x0.clone();
    let dt = 1.0f32 / steps as f32;
    let mut arc = 0.0f32;

    for s in 0..steps {
        let t = (s as f32) * dt;
        let v = field.eval(&x.view(), t, yj);
        let mut x_next = x.clone();
        for k in 0..x.len() {
            x_next[k] = x[k] + dt * v[k];
        }
        arc += l2(&x, &x_next);
        x = x_next;
    }
    let chord_len = l2(x0, &x);
    (arc, chord_len)
}

/// Paper-aligned structural test:
/// rectified (paired) training should improve *few-step* behavior versus a baseline
/// that uses independent/random coupling.
///
/// We measure:
/// - a sanity straightness proxy (arc/chord should be close to 1 in this small regime), and
/// - a paper-aligned proxy for “straight enough for very low NFE”:
///   1-step integration should land closer to the conditioned target than the baseline.
#[test]
fn rfm_usgs_few_step_behavior_beats_random_coupling() -> Result<()> {
    // Parse USGS
    let mut pts: Vec<[f32; 3]> = Vec::new();
    let mut mags: Vec<f32> = Vec::new();
    for (line_idx, line) in USGS_CSV.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        let lat: f32 = parts[1].parse().unwrap_or(0.0);
        let lon: f32 = parts[2].parse().unwrap_or(0.0);
        let mag: f32 = parts[4].parse().unwrap_or(0.0);
        if !lat.is_finite() || !lon.is_finite() || !mag.is_finite() {
            continue;
        }
        pts.push(latlon_to_unit_xyz(lat, lon));
        mags.push(mag);
    }
    assert!(pts.len() >= 10);
    let n = pts.len();

    let d = 3usize;
    let mut y = Array2::<f32>::zeros((n, d));
    for (i, p) in pts.iter().enumerate() {
        y[[i, 0]] = p[0];
        y[[i, 1]] = p[1];
        y[[i, 2]] = p[2];
    }
    let mut b = Array1::<f32>::zeros(n);
    for i in 0..n {
        b[i] = (mags[i] - 5.0).max(0.0).exp();
    }

    // Train settings (keep runtime bounded).
    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: 900,
        batch_size: 64,
        sample_steps: 0,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };

    // RFM (paired).
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 0.2,
        max_iter: 800,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 2,
    };
    let rfm = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;

    // Baseline: random coupling via categorical assignment (no rectification).
    // Potentials are not used for this assignment strategy; keep it cheap.
    let pot_cfg = SemidiscreteSgdConfig {
        epsilon: 0.0,
        lr: 0.1,
        steps: 1,
        batch_size: 1,
        seed: 0,
    };
    let rand_coupling = train_sd_fm_semidiscrete_linear_with_assignment(
        &y.view(),
        &(&b / b.sum()).view(),
        &pot_cfg,
        &fm_cfg,
        SdFmTrainAssignment::CategoricalFromB,
    )?;

    // Compare straightness and 1-step error across shared x0 stream.
    let n_samp = 128usize;
    let seed = 777u64;
    let steps_path = 64usize;
    let (x0s, x1_rfm_1, js) = rfm.sample_with_x0(n_samp, seed, 1)?;
    let (_x0s2, x1_base_1, js2) = rand_coupling.sample_with_x0(n_samp, seed, 1)?;
    assert_eq!(
        js, js2,
        "categorical assignment should match under identical RNG stream"
    );

    let mut sum_rfm = 0.0f64;
    let mut sum_base = 0.0f64;
    let mut mse_rfm = 0.0f64;
    let mut mse_base = 0.0f64;
    let mut count = 0usize;

    for i in 0..n_samp {
        let x0 = x0s.row(i).to_owned();
        let j = js[i];
        let yj = y.row(j);

        let (arc_r, chord_r) = integrate_euler_path(&x0, &yj, &rfm.field, steps_path);
        let (arc_b, chord_b) = integrate_euler_path(&x0, &yj, &rand_coupling.field, steps_path);

        if chord_r > 1e-6 && chord_b > 1e-6 {
            sum_rfm += (arc_r / chord_r) as f64;
            sum_base += (arc_b / chord_b) as f64;

            // 1-step “land near the conditioned target” error
            let mut e_r = 0.0f64;
            let mut e_b = 0.0f64;
            for k in 0..d {
                let dr = (x1_rfm_1[[i, k]] - yj[k]) as f64;
                let db = (x1_base_1[[i, k]] - yj[k]) as f64;
                e_r += dr * dr;
                e_b += db * db;
            }
            mse_rfm += e_r / d as f64;
            mse_base += e_b / d as f64;
            count += 1;
        }
    }

    assert!(count > 32);
    let mean_r = (sum_rfm / count as f64) as f32;
    let mean_b = (sum_base / count as f64) as f32;
    let mse_r = (mse_rfm / count as f64) as f32;
    let mse_b = (mse_base / count as f64) as f32;

    // Sanity: these small learned flows should not be wildly curvy.
    assert!(
        mean_r < 1.05 && mean_b < 1.05,
        "unexpectedly curvy paths: rfm={mean_r:.4} base={mean_b:.4}"
    );

    // Paper-aligned few-step proxy: paired/rectified training should help 1-step behavior.
    assert!(
        mse_r < 0.98 * mse_b,
        "expected RFM lower 1-step MSE to conditioned target: rfm={mse_r:.4} baseline={mse_b:.4} (arc/chord rfm={mean_r:.4} base={mean_b:.4})"
    );

    Ok(())
}
