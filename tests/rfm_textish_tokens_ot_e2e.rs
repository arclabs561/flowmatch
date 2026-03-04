use flowmatch::metrics::ot_cost_samples_to_weighted_support;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

#[path = "../examples/common/textish.rs"]
mod textish;

#[test]
fn rfm_textish_tokens_reduces_ot_cost_vs_zero_field() {
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
        .map(textish::normalize_token)
        .filter(|t| !t.is_empty())
        .collect();
    let toks_b_raw: Vec<String> = doc_b
        .split_whitespace()
        .map(textish::normalize_token)
        .filter(|t| !t.is_empty())
        .collect();

    let bow_a = textish::bag_of_tokens(&toks_a_raw);
    let bow_b = textish::bag_of_tokens(&toks_b_raw);
    let toks_b: Vec<String> = bow_b.iter().map(|(t, _)| t.clone()).collect();
    let (_w_a, w_b) = textish::weights_tfidf_2docs(&bow_a, &bow_b);

    let dim = 48usize;
    let n = toks_b.len();
    let d = dim;

    let mut y = Array2::<f32>::zeros((n, d));
    for (j, tok) in toks_b.iter().enumerate() {
        let v = textish::embed_char_ngrams_signed(tok, dim);
        for k in 0..d {
            y[[j, k]] = v[k];
        }
    }

    let rfm_cfg = RfmMinibatchOtConfig {
        reg: 1.5,
        max_iter: 6000,
        tol: 2e-3,
        pairing: RfmMinibatchPairing::SinkhornGreedy,
        pairing_every: 4,
    };
    let fm_cfg = SdFmTrainConfig {
        lr: 1.0e-2,
        steps: 250,
        batch_size: 64,
        sample_steps: 25,
        seed: 29,
        t_schedule: TimestepSchedule::Uniform,
    };

    let trained = train_rfm_minibatch_ot_linear(&y.view(), &w_b.view(), &rfm_cfg, &fm_cfg).unwrap();

    // Evaluate OT cost for trained samples.
    let (xs, _js) = trained.sample(192, 777, fm_cfg.sample_steps).unwrap();
    let d_tr = ot_cost_samples_to_weighted_support(
        &xs.view(),
        &trained.y.view(),
        &trained.b.view(),
        0.10,
        8000,
        2e-3,
    )
    .unwrap();

    // Baseline: samples from noise (x0) without any flow.
    let mut rng = ChaCha8Rng::seed_from_u64(777);
    let mut x0s = Array2::<f32>::zeros((192, d));
    for i in 0..192 {
        for k in 0..d {
            x0s[[i, k]] = StandardNormal.sample(&mut rng);
        }
    }
    let d_0 = ot_cost_samples_to_weighted_support(
        &x0s.view(),
        &trained.y.view(),
        &trained.b.view(),
        0.10,
        8000,
        2e-3,
    )
    .unwrap();

    assert!(
        d_tr < 0.85 * d_0,
        "expected OT cost improvement: trained={d_tr:.4} baseline={d_0:.4}"
    );
}
