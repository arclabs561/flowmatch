//! Protein torsion angle helpers shared across multiple examples.
//!
//! Provides wrap-to-pi, torus embedding/decoding, Ramachandran histogram, and CSV parsing.

use ndarray::{Array1, Array2};

/// Vendored phi/psi angles from PDB 1BPI.
pub const PHI_PSI_CSV: &str = include_str!("../../examples_data/pdb_1bpi_phi_psi.csv.txt");

/// Map angle to (-pi, pi].
pub fn wrap_pi(x: f32) -> f32 {
    let mut y = x % (2.0 * core::f32::consts::PI);
    if y <= -core::f32::consts::PI {
        y += 2.0 * core::f32::consts::PI;
    }
    if y > core::f32::consts::PI {
        y -= 2.0 * core::f32::consts::PI;
    }
    y
}

/// Embed (phi, psi) into R^4 via (cos phi, sin phi, cos psi, sin psi).
pub fn embed_phi_psi(phi: f32, psi: f32) -> [f32; 4] {
    [phi.cos(), phi.sin(), psi.cos(), psi.sin()]
}

/// Decode a 4D embedding back to (phi, psi) angles via atan2.
pub fn decode_phi_psi(e: [f32; 4]) -> (f32, f32) {
    let phi = e[1].atan2(e[0]);
    let psi = e[3].atan2(e[2]);
    (wrap_pi(phi), wrap_pi(psi))
}

/// Compute a 2D Ramachandran histogram (bins x bins), normalized to sum 1.
pub fn rama_hist(phi_psi: &[(f32, f32)], bins: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; bins * bins];
    let two_pi = 2.0 * core::f32::consts::PI;
    for &(phi, psi) in phi_psi {
        let u = (wrap_pi(phi) + core::f32::consts::PI) / two_pi; // [0,1)
        let v = (wrap_pi(psi) + core::f32::consts::PI) / two_pi; // [0,1)
        let mut i = (u * bins as f32).floor() as isize;
        let mut j = (v * bins as f32).floor() as isize;
        if i < 0 {
            i = 0;
        }
        if j < 0 {
            j = 0;
        }
        if i >= bins as isize {
            i = bins as isize - 1;
        }
        if j >= bins as isize {
            j = bins as isize - 1;
        }
        h[(i as usize) * bins + (j as usize)] += 1.0;
    }
    let sum: f32 = h.iter().sum();
    if sum > 0.0 {
        for x in &mut h {
            *x /= sum;
        }
    }
    h
}

/// Parse the vendored phi/psi CSV (6-column format: ..., phi, psi).
///
/// Returns `Err` if fewer than `min_pairs` valid rows are found.
pub fn parse_phi_psi_csv_6col(min_pairs: usize) -> flowmatch::Result<Vec<(f32, f32)>> {
    let mut phi_psi: Vec<(f32, f32)> = Vec::new();
    for (line_idx, line) in PHI_PSI_CSV.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        let phi: f32 = parts[4].parse().unwrap_or(0.0);
        let psi: f32 = parts[5].parse().unwrap_or(0.0);
        if phi.is_finite() && psi.is_finite() {
            phi_psi.push((phi, psi));
        }
    }
    if phi_psi.len() < min_pairs {
        return Err(flowmatch::Error::Domain("not enough parsed phi/psi pairs"));
    }
    Ok(phi_psi)
}

/// Parse the vendored phi/psi CSV (2-column format: phi, psi).
///
/// Returns `Err` if fewer than `min_pairs` valid rows are found.
pub fn parse_phi_psi_csv_2col(min_pairs: usize) -> flowmatch::Result<Vec<(f32, f32)>> {
    let mut phi_psi: Vec<(f32, f32)> = Vec::new();
    for (line_idx, line) in PHI_PSI_CSV.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        let phi: f32 = parts[0].parse().unwrap_or(0.0);
        let psi: f32 = parts[1].parse().unwrap_or(0.0);
        if phi.is_finite() && psi.is_finite() {
            phi_psi.push((wrap_pi(phi), wrap_pi(psi)));
        }
    }
    if phi_psi.len() < min_pairs {
        return Err(flowmatch::Error::Domain("not enough torsion points parsed"));
    }
    Ok(phi_psi)
}

/// Build the discrete support array `y` (n x 4) and uniform weight vector from parsed torsion data.
pub fn build_torsion_support(phi_psi: &[(f32, f32)]) -> (Array2<f32>, Array1<f32>) {
    let n = phi_psi.len();
    let d = 4usize;
    let mut y = Array2::<f32>::zeros((n, d));
    for (i, &(phi, psi)) in phi_psi.iter().enumerate() {
        let e = embed_phi_psi(phi, psi);
        for k in 0..d {
            y[[i, k]] = e[k];
        }
    }
    let b = Array1::<f32>::from_elem(n, 1.0);
    (y, b)
}
