//! Demonstrate minibatch OT "outlier forcing" and a partial pairing fix.
//!
//! In full one-to-one pairing, every target column must be used. If a minibatch contains a rare
//! outlier target, some source sample will be forced to match it, creating a huge displacement.
//!
//! Partial pairing (one-to-one only for the easiest rows, then NN fallback for the rest) avoids
//! forcing the outlier to be used.
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_minibatch_outlier_partial
//! ```

use flowmatch::rfm::{
    minibatch_ot_greedy_pairing, minibatch_ot_selective_pairing, minibatch_partial_rowwise_pairing,
    minibatch_rowwise_nearest_pairing,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 64usize;
    let d = 8usize;
    let mut rng = ChaCha8Rng::seed_from_u64(123);

    let mut x = Array2::<f32>::zeros((n, d));
    let mut y = Array2::<f32>::zeros((n, d));
    for i in 0..n {
        for k in 0..d {
            x[[i, k]] = StandardNormal.sample(&mut rng);
            y[[i, k]] = StandardNormal.sample(&mut rng);
        }
    }

    // Inject a rare outlier in y (very far from everything).
    for k in 0..d {
        y[[n - 1, k]] = 100.0;
    }

    let keep_frac: f32 = std::env::var("FLOWMATCH_PAIRING_PARTIAL_KEEP_FRAC")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.8);

    let perm_full = minibatch_rowwise_nearest_pairing(&x.view(), &y.view())?;
    let perm_part = minibatch_partial_rowwise_pairing(&x.view(), &y.view(), keep_frac)?;

    // Also show Sinkhorn-based variants (slower, but more "RFM-faithful").
    let perm_sinkhorn = minibatch_ot_greedy_pairing(&x.view(), &y.view(), 0.2, 2_000, 2e-3)?;
    let perm_sinkhorn_sel =
        minibatch_ot_selective_pairing(&x.view(), &y.view(), 0.2, 2_000, 2e-3, keep_frac)?;

    fn summarize(name: &str, x: &Array2<f32>, y: &Array2<f32>, perm: &[usize]) {
        let n = x.nrows();
        let d = x.ncols();
        let mut used_outlier = 0usize;
        let mut max_d = 0.0f64;
        for i in 0..n {
            let j = perm[i];
            if j == n - 1 {
                used_outlier += 1;
            }
            let mut s = 0.0f64;
            for k in 0..d {
                let dk = (x[[i, k]] - y[[j, k]]) as f64;
                s += dk * dk;
            }
            max_d = max_d.max(s.sqrt());
        }
        println!("- {name}: used_outlier_col={used_outlier}  max_pair_dist={max_d:.3}");
    }

    println!("minibatch outlier demo (n={n}, d={d})");
    println!("- keep_frac={keep_frac}");
    summarize("rowwise_one_to_one", &x, &y, &perm_full);
    summarize("partial_rowwise", &x, &y, &perm_part);
    summarize("sinkhorn_greedy", &x, &y, &perm_sinkhorn);
    summarize("sinkhorn_selective", &x, &y, &perm_sinkhorn_sel);

    Ok(())
}
