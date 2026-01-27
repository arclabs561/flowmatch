//! Rectified flow matching on **real geospatial data** (earthquakes on the sphere).
//!
//! This example is motivated by “Flow Matching on General Geometries” style benchmarks where
//! *sphere-valued* data (lat/lon) is the native object. We stay within `flowmatch`’s current
//! constraints (a tiny linear conditional field in Euclidean space) but:
//!
//! - we use a **real dataset** (USGS earthquake catalog),
//! - we evaluate with a **distributional metric** (entropic OT cost to a weighted support),
//! - we keep determinism explicit (seeded).
//!
//! Data provenance:
//! - Source: USGS Earthquake Catalog API (CSV)
//! - Query used (limit 50, M≥6, year 2024):
//!   `https://earthquake.usgs.gov/fdsnws/event/1/query.csv?format=csv&starttime=2024-01-01&endtime=2024-12-31&minmagnitude=6&orderby=time&limit=50`
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_usgs_earthquakes_sphere
//! ```

use flowmatch::metrics::ot_cost_samples_to_weighted_support;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
    TimestepSchedule,
};
use flowmatch::Result;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

// Pulled from the USGS CSV endpoint above (a small, fixed slice; no network needed at runtime).
const USGS_CSV: &str = include_str!("../examples_data/usgs_eq_m6_2024_limit50.csv.txt");

fn deg_to_rad(x: f32) -> f32 {
    x * core::f32::consts::PI / 180.0
}

fn rad_to_deg(x: f32) -> f32 {
    x * 180.0 / core::f32::consts::PI
}

fn latlon_to_unit_xyz(lat_deg: f32, lon_deg: f32) -> [f32; 3] {
    let lat = deg_to_rad(lat_deg);
    let lon = deg_to_rad(lon_deg);
    let clat = lat.cos();
    [
        clat * lon.cos(),
        clat * lon.sin(),
        lat.sin(), // z
    ]
}

fn unit_xyz_to_latlon(v: [f32; 3]) -> (f32, f32) {
    let (x, y, z) = (v[0], v[1], v[2]);
    let zc = z.clamp(-1.0, 1.0);
    let lat = zc.asin();
    let lon = y.atan2(x);
    (rad_to_deg(lat), rad_to_deg(lon))
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 0.0 && n.is_finite() {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn main() -> Result<()> {
    // Parse: we only need the early columns, before the quoted `place` field, so simple splitting
    // is safe here.
    let mut pts: Vec<[f32; 3]> = Vec::new();
    let mut mags: Vec<f32> = Vec::new();

    for (line_idx, line) in USGS_CSV.lines().enumerate() {
        if line_idx == 0 {
            continue;
        }
        if line.trim().is_empty() {
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

    if pts.len() < 10 {
        return Err(flowmatch::Error::Domain("not enough parsed USGS points"));
    }

    let n = pts.len();
    let d = 3usize;

    // Discrete support y_j are the observed points on S^2 embedded in R^3.
    let mut y = Array2::<f32>::zeros((n, d));
    for (i, p) in pts.iter().enumerate() {
        y[[i, 0]] = p[0];
        y[[i, 1]] = p[1];
        y[[i, 2]] = p[2];
    }

    // Weight earthquakes slightly toward higher magnitudes.
    // (No hidden normalization: we explicitly normalize before training.)
    let mut b = Array1::<f32>::zeros(n);
    for i in 0..n {
        b[i] = (mags[i] - 5.0).max(0.0).exp(); // >= 1.0 for larger quakes; stable.
    }

    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: 2_500,
        batch_size: 64,
        sample_steps: 30,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };
    let rfm_cfg = RfmMinibatchOtConfig {
        // Slightly “easier” coupling settings to reduce convergence failures in small minibatches.
        reg: 0.2,
        max_iter: 2_000,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 1,
    };

    // Baseline: raw Gaussian noise (no learned flow), then project to the sphere.
    let baseline = {
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let mut xs = Array2::<f32>::zeros((256, d));
        for i in 0..xs.nrows() {
            let v = normalize3([
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ]);
            xs[[i, 0]] = v[0];
            xs[[i, 1]] = v[1];
            xs[[i, 2]] = v[2];
        }
        ot_cost_samples_to_weighted_support(&xs.view(), &y.view(), &b.view(), 0.05, 800, 1e-4)?
    };

    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;

    let (xs_raw, _js) = model.sample(256, 777, fm_cfg.sample_steps)?;
    let mut xs = xs_raw.clone();
    for i in 0..xs.nrows() {
        let v = normalize3([xs[[i, 0]], xs[[i, 1]], xs[[i, 2]]]);
        xs[[i, 0]] = v[0];
        xs[[i, 1]] = v[1];
        xs[[i, 2]] = v[2];
    }

    let trained =
        ot_cost_samples_to_weighted_support(&xs.view(), &y.view(), &b.view(), 0.05, 800, 1e-4)?;

    println!("USGS earthquakes (n={n}), embedding=R^3 with S^2 projection");
    println!("OT cost (lower is better):");
    println!("- baseline (near-noise): {baseline:.4}");
    println!("- trained  (RFM+minibatch OT): {trained:.4}");
    println!("- ratio trained/baseline: {:.3}", trained / baseline);

    println!("\nSome generated samples (lat, lon):");
    for i in 0..8.min(xs.nrows()) {
        let (lat, lon) = unit_xyz_to_latlon([xs[[i, 0]], xs[[i, 1]], xs[[i, 2]]]);
        println!("  {i:>2}: lat={lat:>7.2}°, lon={lon:>8.2}°");
    }

    Ok(())
}
