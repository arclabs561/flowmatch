//! Shared helpers for flowmatch examples.
//!
//! This module exists to de-duplicate code across the USGS earthquake and protein
//! torsion examples. It is not part of the public API.

#[allow(dead_code)]
pub mod textish;
#[allow(dead_code)]
pub mod torsions;
#[allow(dead_code)]
pub mod usgs;

#[allow(dead_code)]
/// Mean squared distance from each sample to its assigned target.
pub fn mean_sq_to_assigned_y(
    xs: &ndarray::Array2<f32>,
    js: &[usize],
    y: &ndarray::Array2<f32>,
) -> f32 {
    let n = xs.nrows();
    let d = xs.ncols();
    let mut s: f64 = 0.0;
    for i in 0..n {
        let j = js[i];
        for k in 0..d {
            let r = (xs[[i, k]] - y[[j, k]]) as f64;
            s += r * r;
        }
    }
    (s / (n as f64 * d as f64)) as f32
}

/// Timing breakdown for profile examples (USGS, torsions).
///
/// Each field accumulates wall-clock time for a phase of the training loop.
#[derive(Default, Debug, Clone)]
#[allow(dead_code)]
pub struct Timings {
    pub sample_x0: std::time::Duration,
    pub sample_y: std::time::Duration,
    pub sinkhorn_pair: std::time::Duration,
    pub fast_pair: std::time::Duration,
    pub sgd: std::time::Duration,
}

#[allow(dead_code)]
impl Timings {
    /// Print a formatted timing breakdown. `fast_pair` is excluded from the
    /// accounted total (it is extra profiling work, not part of training).
    pub fn print(&self) {
        let total = self.sample_x0 + self.sample_y + self.sinkhorn_pair + self.sgd;
        let denom = total.as_secs_f64().max(1e-12);
        for (name, dur) in [
            ("sample_x0", self.sample_x0),
            ("sample_y", self.sample_y),
            ("sinkhorn_pair", self.sinkhorn_pair),
            ("fast_pair", self.fast_pair),
            ("sgd", self.sgd),
        ] {
            println!(
                "  - {:>12}: {:>10.3} ms ({:>5.1}%)",
                name,
                1e3 * dur.as_secs_f64(),
                100.0 * dur.as_secs_f64() / denom
            );
        }
    }

    /// Accounted total (excludes `fast_pair`).
    pub fn accounted_total(&self) -> std::time::Duration {
        self.sample_x0 + self.sample_y + self.sinkhorn_pair + self.sgd
    }
}

#[allow(dead_code)]
/// Mean and (population) standard deviation of a slice.
pub fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len().max(1) as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

#[allow(dead_code)]
/// Community-size distribution (descending, zero-padded to `top_k` entries, normalized to sum 1).
///
/// Used by the kNN+Leiden examples. Lives here because it is shared across multiple USGS examples.
pub fn community_size_distribution(labels: &[usize], top_k: usize) -> Vec<f32> {
    use std::collections::HashMap;
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &c in labels {
        *counts.entry(c).or_insert(0) += 1;
    }
    let mut sizes: Vec<f32> = counts.values().map(|&x| x as f32).collect();
    sizes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = sizes.iter().sum();
    if total > 0.0 {
        for x in &mut sizes {
            *x /= total;
        }
    }
    sizes.truncate(top_k);
    while sizes.len() < top_k {
        sizes.push(0.0);
    }
    sizes
}
