//! Shared helpers for flowmatch examples.
//!
//! This module exists to de-duplicate code across the USGS earthquake and protein
//! torsion examples. It is not part of the public API.

#[allow(dead_code)]
pub mod torsions;
#[allow(dead_code)]
pub mod usgs;

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
