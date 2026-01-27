use flowmatch::metrics::ot_cost_samples_to_weighted_support;
use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{
    train_rfm_minibatch_ot_linear, RfmMinibatchOtConfig, RfmMinibatchPairing, SdFmTrainConfig,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

mod textish {
    use ndarray::Array1;
    use std::collections::BTreeMap;

    pub fn normalize_token(tok: &str) -> String {
        let mut out = String::new();
        for c in tok.chars() {
            if c.is_alphanumeric() {
                out.extend(c.to_lowercase());
            }
        }
        out
    }

    fn base_weight(tok: &str) -> f32 {
        if tok.is_empty() {
            return 0.0;
        }
        if tok.len() <= 3 {
            return 0.05;
        }
        if matches!(
            tok,
            "subscribe"
                | "newsletter"
                | "cookies"
                | "cookie"
                | "privacy"
                | "policy"
                | "terms"
                | "menu"
                | "home"
                | "contact"
                | "share"
                | "twitter"
                | "confidential"
                | "internal"
        ) {
            return 0.05;
        }
        1.0
    }

    pub fn bag_of_tokens(tokens: &[String]) -> Vec<(String, usize)> {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for t in tokens {
            if t.is_empty() || t.len() < 2 {
                continue;
            }
            *counts.entry(t.clone()).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    pub fn weights_tfidf_2docs(
        a: &[(String, usize)],
        b: &[(String, usize)],
    ) -> (Array1<f32>, Array1<f32>) {
        let mut df: BTreeMap<&str, usize> = BTreeMap::new();
        for (t, _) in a {
            *df.entry(t.as_str()).or_insert(0) += 1;
        }
        for (t, _) in b {
            *df.entry(t.as_str()).or_insert(0) += 1;
        }

        fn idf(df: usize) -> f32 {
            let n = 2.0f32;
            ((1.0 + n) / (1.0 + df as f32)).ln() + 1.0
        }

        let mut w_a = Array1::<f32>::zeros(a.len());
        for (i, (t, c)) in a.iter().enumerate() {
            let df_t = *df.get(t.as_str()).unwrap_or(&1);
            w_a[i] = base_weight(t) * (*c as f32) * idf(df_t);
        }
        let mut w_b = Array1::<f32>::zeros(b.len());
        for (i, (t, c)) in b.iter().enumerate() {
            let df_t = *df.get(t.as_str()).unwrap_or(&1);
            w_b[i] = base_weight(t) * (*c as f32) * idf(df_t);
        }
        let sa = w_a.sum();
        if sa > 0.0 {
            w_a /= sa;
        }
        let sb = w_b.sum();
        if sb > 0.0 {
            w_b /= sb;
        }
        (w_a, w_b)
    }

    pub fn embed_char_ngrams_signed(text: &str, dim: usize) -> Array1<f32> {
        let mut v = Array1::<f32>::zeros(dim);
        if dim == 0 {
            return v;
        }

        const BOS: u32 = 0x110000;
        const EOS: u32 = 0x110001;
        let mut xs: Vec<u32> = Vec::with_capacity(text.chars().count() + 2);
        xs.push(BOS);
        xs.extend(text.chars().map(|c| c as u32));
        xs.push(EOS);

        fn fnv1a_u32(seq: &[u32]) -> u64 {
            let mut h: u64 = 0xcbf29ce484222325;
            for &x in seq {
                h ^= x as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        }

        for n in [3usize, 2, 1] {
            if xs.len() < n {
                continue;
            }
            for i in 0..=xs.len() - n {
                let h = fnv1a_u32(&xs[i..i + n]);
                let idx = (h as usize) % dim;
                let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
                v[idx] += sign;
            }
        }

        let norm = v.dot(&v).sqrt();
        if norm > 0.0 {
            v /= norm;
        }
        v
    }
}

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
        reg: 0.25,
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
