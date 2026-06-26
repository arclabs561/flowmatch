//! Use fynch regularizers as flowmatch Bregman energy objectives.
//!
//! `flowmatch::energy` is generic over `logp::BregmanGenerator`. With fynch's
//! optional `logp` feature, the exact Fenchel-Young overlaps (`Shannon` and
//! `SquaredL2`) satisfy that boundary directly.

use flowmatch::energy::{bregman_energy_gradient, bregman_energy_target};
use ndarray::arr1;

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn main() {
    let center = arr1(&[0.2, 0.3, 0.5]);
    let x = arr1(&[0.1, 0.4, 0.5]);
    let sigma = 0.5;

    let l2_energy = bregman_energy_target(&fynch::SquaredL2, &x.view(), &center.view(), sigma)
        .expect("squared-L2 energy");
    let logp_l2_energy = bregman_energy_target(&logp::SquaredL2, &x.view(), &center.view(), sigma)
        .expect("logp squared-L2 energy");
    let shannon_energy = bregman_energy_target(&fynch::Shannon, &x.view(), &center.view(), sigma)
        .expect("Shannon energy");
    let shannon_grad = bregman_energy_gradient(&fynch::Shannon, &x.view(), &center.view(), sigma)
        .expect("Shannon gradient");

    println!("x: {x:?}");
    println!("center: {center:?}");
    println!("squared-L2 energy via fynch: {l2_energy:.6}");
    println!("squared-L2 energy via logp:  {logp_l2_energy:.6}");
    println!("Shannon energy via fynch:    {shannon_energy:.6}");
    println!(
        "Shannon gradient norm:       {:.6}",
        l2(shannon_grad.as_slice().unwrap(), &[0.0; 3])
    );

    assert!((l2_energy - logp_l2_energy).abs() < 1e-7);
    assert!(shannon_energy < 0.0);
    assert!(shannon_grad.iter().all(|v| v.is_finite()));
}
