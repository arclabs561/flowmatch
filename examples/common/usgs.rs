//! USGS earthquake data helpers shared across multiple examples.
//!
//! Provides CSV parsing, coordinate conversion, sphere projection, weight
//! construction, and baseline sampling. The vendored CSV is embedded at compile
//! time via `include_str!`.

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Vendored USGS earthquake catalog (M>=6, 2024, limit 50).
pub const USGS_CSV: &str = include_str!("../../examples_data/usgs_eq_m6_2024_limit50.csv.txt");

pub fn deg_to_rad(x: f32) -> f32 {
    x * core::f32::consts::PI / 180.0
}

pub fn rad_to_deg(x: f32) -> f32 {
    x * 180.0 / core::f32::consts::PI
}

pub fn latlon_to_unit_xyz(lat_deg: f32, lon_deg: f32) -> [f32; 3] {
    let lat = deg_to_rad(lat_deg);
    let lon = deg_to_rad(lon_deg);
    let clat = lat.cos();
    [clat * lon.cos(), clat * lon.sin(), lat.sin()]
}

pub fn unit_xyz_to_latlon(v: [f32; 3]) -> (f32, f32) {
    let (x, y, z) = (v[0], v[1], v[2]);
    let zc = z.clamp(-1.0, 1.0);
    let lat = zc.asin();
    let lon = y.atan2(x);
    (rad_to_deg(lat), rad_to_deg(lon))
}

pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 0.0 && n.is_finite() {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        [1.0, 0.0, 0.0]
    }
}

/// Assign a point to the nearest center by cosine similarity (dot product on unit vectors).
pub fn argmax_dot_unit(v: [f32; 3], centers: &[[f32; 3]]) -> usize {
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, c) in centers.iter().enumerate() {
        let s = v[0] * c[0] + v[1] * c[1] + v[2] * c[2];
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    best
}

/// Cosine distance between two unit vectors.
pub fn cosine_distance_unit(a: [f32; 3], b: [f32; 3]) -> f32 {
    1.0 - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
}

/// Parsed USGS earthquake data: unit-sphere XYZ points and magnitudes.
pub struct UsgsData {
    pub pts: Vec<[f32; 3]>,
    pub mags: Vec<f32>,
}

/// Parse the vendored USGS CSV into XYZ points and magnitudes.
///
/// Returns `Err` if fewer than `min_points` valid rows are found.
pub fn parse_usgs_csv(min_points: usize) -> flowmatch::Result<UsgsData> {
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
    if pts.len() < min_points {
        return Err(flowmatch::Error::Domain("not enough parsed USGS points"));
    }
    Ok(UsgsData { pts, mags })
}

/// Build the discrete support array `y` (n x 3) and magnitude-derived weight vector `b`
/// from parsed USGS data.
pub fn build_support_and_weights(data: &UsgsData) -> (Array2<f32>, Array1<f32>) {
    let n = data.pts.len();
    let d = 3usize;
    let mut y = Array2::<f32>::zeros((n, d));
    for (i, p) in data.pts.iter().enumerate() {
        y[[i, 0]] = p[0];
        y[[i, 1]] = p[1];
        y[[i, 2]] = p[2];
    }
    let mut b = Array1::<f32>::zeros(n);
    for i in 0..n {
        b[i] = (data.mags[i] - 5.0).max(0.0).exp();
    }
    (y, b)
}

/// Generate baseline samples: Gaussian noise projected to the unit sphere.
pub fn baseline_sphere_samples(n_samples: usize, d: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut xs = Array2::<f32>::zeros((n_samples, d));
    for i in 0..n_samples {
        let v = normalize3([
            StandardNormal.sample(&mut rng),
            StandardNormal.sample(&mut rng),
            StandardNormal.sample(&mut rng),
        ]);
        xs[[i, 0]] = v[0];
        xs[[i, 1]] = v[1];
        xs[[i, 2]] = v[2];
    }
    xs
}

/// Project an array of R^3 points onto the unit sphere (in-place normalization).
pub fn project_to_sphere(xs: &mut Array2<f32>) {
    for i in 0..xs.nrows() {
        let v = normalize3([xs[[i, 0]], xs[[i, 1]], xs[[i, 2]]]);
        xs[[i, 0]] = v[0];
        xs[[i, 1]] = v[1];
        xs[[i, 2]] = v[2];
    }
}

/// Build an exact kNN graph using cosine distance on unit vectors.
///
/// Returns an undirected petgraph with edge weight = max(0.001, 1 - cosine_distance).
pub fn exact_knn_graph(points: &[[f32; 3]], k: usize) -> petgraph::graph::UnGraph<(), f32> {
    use petgraph::graph::UnGraph;
    let n = points.len();
    let k = k.min(n.saturating_sub(1));
    let mut g = UnGraph::<(), f32>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();

    for i in 0..n {
        let mut dists: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_distance_unit(points[i], points[j])))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(j, dist) in dists.iter().take(k) {
            let w = (1.0 - dist).max(0.001);
            let _ = g.add_edge(nodes[i], nodes[j], w);
        }
    }
    g
}
