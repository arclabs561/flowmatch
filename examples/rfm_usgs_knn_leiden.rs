//! Full-pipeline structure demo (flowmatch + wass + parti + jin):
//!
//! - Train RFM on real USGS earthquake locations (sphere in R^3).
//! - Sample points from the learned flow.
//! - Build a kNN graph using `parti`'s `knn_graph_with_config` (HNSW via `jin`).
//! - Run Leiden community detection and compare **community-size distributions**
//!   against the real-data graph via JS divergence.
//!
//! This is meant as a "how the engine composes" example, not a perfectly deterministic test:
//! HNSW construction involves randomness internally.
//!
//! Run:
//! ```bash
//! cargo run -p flowmatch --example rfm_usgs_knn_leiden
//! ```

mod common;

use common::community_size_distribution;
use common::usgs::{build_support_and_weights, normalize3, parse_usgs_csv};
use flowmatch::metrics::jensen_shannon_divergence_histogram;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use flowmatch::Result;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use parti::community::CommunityDetection;
use parti::{knn_graph_with_config, KnnGraphConfig, Leiden, WeightFunction};

fn main() -> Result<()> {
    let data = parse_usgs_csv(10)?;
    let n = data.pts.len();
    let (y, b) = build_support_and_weights(&data);

    // Build real kNN graph + Leiden labels.
    let embeddings_real: Vec<Vec<f32>> = data.pts.iter().map(|p| vec![p[0], p[1], p[2]]).collect();
    let knn_cfg = KnnGraphConfig {
        k: 10.min(n.saturating_sub(1)),
        symmetric: true,
        weight_fn: WeightFunction::InverseDistance,
        ..Default::default()
    };
    let graph_real = knn_graph_with_config(&embeddings_real, &knn_cfg)
        .map_err(|_| flowmatch::Error::Domain("parti::knn_graph_with_config failed"))?;
    let leiden = Leiden::new().with_resolution(1.0).with_seed(42);
    let labels_real = leiden
        .detect(&graph_real)
        .map_err(|_| flowmatch::Error::Domain("Leiden failed on real graph"))?;
    let real_sizes = community_size_distribution(&labels_real, 16);

    // Baseline: random-on-sphere samples (no learning), then kNN+Leiden.
    let baseline_js = {
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let mut emb: Vec<Vec<f32>> = Vec::new();
        for _ in 0..400 {
            let v = normalize3([
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ]);
            emb.push(vec![v[0], v[1], v[2]]);
        }
        let g = knn_graph_with_config(&emb, &knn_cfg)
            .map_err(|_| flowmatch::Error::Domain("parti knn_graph failed (baseline)"))?;
        let lab = leiden
            .detect(&g)
            .map_err(|_| flowmatch::Error::Domain("Leiden failed (baseline)"))?;
        let sizes = community_size_distribution(&lab, 16);
        jensen_shannon_divergence_histogram(&real_sizes, &sizes, 1e-6)?
    };

    // Train + sample.
    let fm_cfg = SdFmTrainConfig {
        lr: 2e-2,
        steps: 2_500,
        batch_size: 64,
        sample_steps: 30,
        seed: 123,
        t_schedule: TimestepSchedule::Uniform,
    };
    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 1.0,
        max_iter: 2_000,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 1,
    };
    let model = train_rfm_minibatch_ot_linear(&y.view(), &b.view(), &rfm_cfg, &fm_cfg)?;
    let (xs_raw, _js) = model.sample(400, 777, fm_cfg.sample_steps)?;

    let trained_js = {
        let mut emb: Vec<Vec<f32>> = Vec::new();
        for i in 0..xs_raw.nrows() {
            let v = normalize3([xs_raw[[i, 0]], xs_raw[[i, 1]], xs_raw[[i, 2]]]);
            emb.push(vec![v[0], v[1], v[2]]);
        }
        let g = knn_graph_with_config(&emb, &knn_cfg)
            .map_err(|_| flowmatch::Error::Domain("parti knn_graph failed (trained)"))?;
        let lab = leiden
            .detect(&g)
            .map_err(|_| flowmatch::Error::Domain("Leiden failed (trained)"))?;
        let sizes = community_size_distribution(&lab, 16);
        jensen_shannon_divergence_histogram(&real_sizes, &sizes, 1e-6)?
    };

    println!("USGS kNN+Leiden structure eval (non-deterministic HNSW build)");
    println!("JS(community-size(real), community-size(samples)) lower is better:");
    println!("- baseline: {baseline_js:.4}");
    println!("- trained : {trained_js:.4}");
    println!("- ratio  : {:.3}", trained_js / baseline_js);

    Ok(())
}
