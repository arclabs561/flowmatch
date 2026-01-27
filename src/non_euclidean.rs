//! Non-Euclidean (geodesic) FM scaffolding.
//!
//! The survey taxonomy (arXiv:2507.17731) groups “non-Euclidean FM” variants together:
//! the core idea is that your data live on a manifold / curved space, so “straight lines”
//! are replaced by **geodesics**, and the velocity field must respect the geometry.
//!
//! This module provides a tiny abstraction for “geodesic interpolants” so downstream code
//! can be explicit about when it assumes Euclidean structure.

use ndarray::{Array1, ArrayView1};

/// A space in which we can interpolate between two points along a geodesic.
pub trait GeodesicSpace {
    /// Interpolate a point on the geodesic from `p0` to `p1` at time `t in [0,1]`.
    fn geodesic_point(&self, p0: &ArrayView1<f32>, p1: &ArrayView1<f32>, t: f32) -> Array1<f32>;

    /// A reference tangent/velocity target for the geodesic.
    ///
    /// For Euclidean straight lines, this is constant: `p1 - p0`.
    /// For general manifolds, this may depend on `t` (or require log/exp maps).
    fn geodesic_velocity(&self, p0: &ArrayView1<f32>, p1: &ArrayView1<f32>, t: f32) -> Array1<f32>;
}

/// Euclidean space with straight-line geodesics.
#[derive(Debug, Clone, Copy)]
pub struct EuclideanSpace;

impl GeodesicSpace for EuclideanSpace {
    fn geodesic_point(&self, p0: &ArrayView1<f32>, p1: &ArrayView1<f32>, t: f32) -> Array1<f32> {
        let mut out = p0.to_owned();
        for i in 0..out.len() {
            out[i] = (1.0 - t) * p0[i] + t * p1[i];
        }
        out
    }

    fn geodesic_velocity(
        &self,
        p0: &ArrayView1<f32>,
        p1: &ArrayView1<f32>,
        _t: f32,
    ) -> Array1<f32> {
        let mut out = p0.to_owned();
        for i in 0..out.len() {
            out[i] = p1[i] - p0[i];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn euclidean_geodesic_matches_lerp_and_constant_velocity() {
        let p0 = array![1.0f32, -2.0, 0.5];
        let p1 = array![3.0f32, 1.0, -1.5];
        let space = EuclideanSpace;
        let t = 0.25;

        let pt = space.geodesic_point(&p0.view(), &p1.view(), t);
        for i in 0..p0.len() {
            let expected = (1.0 - t) * p0[i] + t * p1[i];
            assert!((pt[i] - expected).abs() < 1e-6);
        }

        let v = space.geodesic_velocity(&p0.view(), &p1.view(), t);
        for i in 0..p0.len() {
            let expected = p1[i] - p0[i];
            assert!((v[i] - expected).abs() < 1e-6);
        }
    }
}
