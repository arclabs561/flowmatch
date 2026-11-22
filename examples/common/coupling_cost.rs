//! Mean squared Euclidean displacement per pair on a minibatch.

use ndarray::{Array2, ArrayView1};

fn squared_distance(x: ArrayView1<'_, f32>, y: ArrayView1<'_, f32>) -> f64 {
    x.iter()
        .zip(y)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum()
}

pub fn pairing_cost(x: &Array2<f32>, y: &Array2<f32>, pairing: &[usize]) -> f64 {
    pairing
        .iter()
        .enumerate()
        .map(|(i, &j)| squared_distance(x.row(i), y.row(j)))
        .sum::<f64>()
        / x.nrows() as f64
}

/// Exact expectation for independent uniform draws from the two empirical batches.
pub fn independent_cost(x: &Array2<f32>, y: &Array2<f32>) -> f64 {
    let mut sum = 0.0;
    for source in x.rows() {
        for target in y.rows() {
            sum += squared_distance(source, target);
        }
    }
    sum / (x.nrows() as f64 * y.nrows() as f64)
}
