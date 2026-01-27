use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use flowmatch::Result;
use ndarray::{Array1, Array2};

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

fn mean_sq_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    assert_eq!(a.dim(), b.dim());
    let mut s: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64) - (*y as f64);
        s += d * d;
    }
    (s / (a.len() as f64)) as f32
}

/// Paper-aligned test: sampling should *converge* as NFE increases.
///
/// We don't assert "quality improves" here (domain metric), we assert numerical self-consistency:
/// the difference between 16-step and 32-step samples should be smaller than the difference
/// between 8-step and 16-step samples (fixed seed, same x0 stream).
#[test]
fn rfm_usgs_sampling_converges_with_more_steps() -> Result<()> {
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

    // Train quickly but non-trivially.
    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: 800,
        batch_size: 64,
        sample_steps: 0,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 0.2,
        max_iter: 800,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 2,
    };
    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;

    let seed = 777u64;
    let n_samp = 512usize;

    let (x0_8, x1_8, _js8) = model.sample_with_x0(n_samp, seed, 8)?;
    let (x0_16, x1_16, _js16) = model.sample_with_x0(n_samp, seed, 16)?;
    let (x0_32, x1_32, _js32) = model.sample_with_x0(n_samp, seed, 32)?;

    // Sanity: same seed should produce identical x0 stream regardless of integration steps.
    assert!(mean_sq_diff(&x0_8, &x0_16) == 0.0);
    assert!(mean_sq_diff(&x0_16, &x0_32) == 0.0);

    let d_8_16 = mean_sq_diff(&x1_8, &x1_16);
    let d_16_32 = mean_sq_diff(&x1_16, &x1_32);

    assert!(
        d_16_32 <= d_8_16,
        "expected convergence with more steps: d(16,32)={d_16_32:.6} d(8,16)={d_8_16:.6}"
    );

    Ok(())
}
