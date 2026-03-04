//! RFM minibatch-OT demo on "text-ish tokens".
//!
//! This is meant to be less toy than random vectors:
//! - build token embeddings (char n-grams)
//! - build TF-IDF-ish token weights from two small documents
//! - treat doc-B tokens as a weighted discrete support
//! - train a flow to generate token embeddings with that weight profile

use flowmatch::metrics::ot_cost_samples_to_weighted_support;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use ndarray::{Array1, Array2};

mod common;

fn topk_indices(w: &Array1<f32>, k: usize) -> Vec<usize> {
    let mut v: Vec<(usize, f32)> = w.iter().copied().enumerate().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(k);
    v.into_iter().map(|(i, _)| i).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let doc_a = r#"
        Breaking: Quarterly earnings show steady growth
        The quarterly earnings showed steady growth in all sectors, with revenue up 12%.
        Subscribe to our newsletter. Cookies policy. Share on Twitter.
    "#;
    let doc_b = r#"
        CONFIDENTIAL - INTERNAL MEMO
        Qarterly earnigns showd stdy grwth across sectrs; revnue +12 percent.
        MENU HOME CONTACT PRIVACY POLICY TERMS
    "#;

    let toks_a_raw: Vec<String> = doc_a
        .split_whitespace()
        .map(common::textish::normalize_token)
        .filter(|t| !t.is_empty())
        .collect();
    let toks_b_raw: Vec<String> = doc_b
        .split_whitespace()
        .map(common::textish::normalize_token)
        .filter(|t| !t.is_empty())
        .collect();

    let bow_a = common::textish::bag_of_tokens(&toks_a_raw);
    let bow_b = common::textish::bag_of_tokens(&toks_b_raw);
    let toks_b: Vec<String> = bow_b.iter().map(|(t, _)| t.clone()).collect();
    let (_w_a, w_b) = common::textish::weights_tfidf_2docs(&bow_a, &bow_b);

    let dim = 64usize;
    let n = toks_b.len();
    let d = dim;

    let mut y = Array2::<f32>::zeros((n, d));
    for (j, tok) in toks_b.iter().enumerate() {
        let v = common::textish::embed_char_ngrams_signed(tok, dim);
        for k in 0..d {
            y[[j, k]] = v[k];
        }
    }

    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 1.5,
        max_iter: 6000,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 1,
    };
    let fm_cfg = SdFmTrainConfig {
        lr: 1.0e-2,
        steps: 400,
        batch_size: 96,
        sample_steps: 30,
        seed: 29,
        t_schedule: TimestepSchedule::Uniform,
    };

    let trained = train_rfm_minibatch_ot_linear(&y.view(), &w_b.view(), &rfm_cfg, &fm_cfg)?;

    let n_samp = 256usize;
    let (xs, _js) = trained.sample(n_samp, 777, fm_cfg.sample_steps)?;

    let ot_cost = ot_cost_samples_to_weighted_support(
        &xs.view(),
        &trained.y.view(),
        &trained.b.view(),
        0.10,
        8000,
        2e-3,
    )?;

    println!("tokens_in_support = {n}  dim = {d}");
    println!(
        "train: steps={} batch={} lr={} seed={}",
        fm_cfg.steps, fm_cfg.batch_size, fm_cfg.lr, fm_cfg.seed
    );
    println!(
        "rfm:   reg={} max_iter={} tol={}",
        rfm_cfg.reg, rfm_cfg.max_iter, rfm_cfg.tol
    );
    println!("eval:  OT_cost(xs -> (y,b)) = {ot_cost:.4}");
    println!();

    let top = topk_indices(&trained.b, 8.min(n));
    println!("Top tokens by target weight:");
    for &j in &top {
        println!("  w={:.4}  tok={}", trained.b[j], toks_b[j]);
    }

    Ok(())
}
