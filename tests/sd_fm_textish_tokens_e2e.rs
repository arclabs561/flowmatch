use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{train_sd_fm_semidiscrete_linear, SdFmTrainConfig};
use ndarray::{Array1, Array2};
use wass::semidiscrete::SemidiscreteSgdConfig;

// A minimal “text-ish” embedding/weighting setup, intentionally mirroring
// `wass/examples/document_alignment_demo.rs` (char n-grams + TF-IDF-ish weights),
// but embedded here so the test is self-contained.
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
        // tiny boilerplate list (enough for the demo strings)
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

fn topk_indices(w: &Array1<f32>, k: usize) -> Vec<usize> {
    let mut v: Vec<(usize, f32)> = w.iter().copied().enumerate().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(k);
    v.into_iter().map(|(i, _)| i).collect()
}

fn hist_from_assignments(js: &[usize], n: usize) -> Array1<f32> {
    let mut h = Array1::<f32>::zeros(n);
    for &j in js {
        h[j] += 1.0;
    }
    let s = h.sum();
    if s > 0.0 {
        h /= s;
    }
    h
}

#[test]
fn sd_fm_textish_token_prototypes_respect_weights() {
    // Mirror `document_alignment_demo.rs` strings (boilerplate + OCR-ish noise).
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

    // Discrete target support y_j := token embeddings for doc B.
    // The semidiscrete potential fit is O(steps × batch × n_tokens × dim).
    // Keep `dim` modest so this stays <10s.
    let dim = 64usize;
    let n = toks_b.len();
    let d = dim;
    let mut y = Array2::<f32>::zeros((n, d));
    for (j, tok) in toks_b.iter().enumerate() {
        let v = textish::embed_char_ngrams_signed(tok, dim);
        for k in 0..d {
            y[[j, k]] = v[k];
        }
    }

    // Train SD-FM to generate these token vectors with frequency w_b.
    let pot_cfg = SemidiscreteSgdConfig {
        epsilon: 0.0,
        lr: 0.8,
        steps: 260,
        batch_size: 192,
        seed: 23,
    };
    // This test is about *assignment frequencies* matching `b`, not about learning a good field.
    // Make FM training essentially a no-op to keep the runtime in check.
    let fm_cfg = SdFmTrainConfig {
        lr: 1.0e-2,
        steps: 1,
        batch_size: 8,
        sample_steps: 20,
        seed: 29,
        t_schedule: TimestepSchedule::Uniform,
    };

    let trained = train_sd_fm_semidiscrete_linear(&y.view(), &w_b.view(), &pot_cfg, &fm_cfg)
        .expect("training should succeed");

    let n_samp = 192usize;
    let (_x0s, _x1s, js) = trained
        .sample_with_x0(n_samp, 777, fm_cfg.sample_steps)
        .expect("sampling should succeed");

    let hist = hist_from_assignments(&js, n);

    // “Realistic” assertion: the high-weight tokens should be selected more often than low-weight tokens.
    let top = topk_indices(&trained.b, 4.min(n));
    let bottom = {
        let mut v: Vec<(usize, f32)> = trained.b.iter().copied().enumerate().collect();
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        v.truncate(4.min(n));
        v.into_iter().map(|(i, _)| i).collect::<Vec<_>>()
    };

    let top_mass: f32 = top.iter().map(|&j| hist[j]).sum();
    let bottom_mass: f32 = bottom.iter().map(|&j| hist[j]).sum();

    assert!(
        top_mass > 1.5 * bottom_mass,
        "expected top tokens to get more assignment mass: top_mass={top_mass:.3} bottom_mass={bottom_mass:.3}"
    );
}
