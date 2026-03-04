use flowmatch::sd_fm::TimestepSchedule;
use flowmatch::sd_fm::{train_sd_fm_semidiscrete_linear, SdFmTrainConfig};
use ndarray::{Array1, Array2};
use wass::semidiscrete::SemidiscreteSgdConfig;

#[path = "../examples/common/textish.rs"]
mod textish;

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

    let pot_cfg = SemidiscreteSgdConfig {
        epsilon: 0.0,
        lr: 0.8,
        steps: 260,
        batch_size: 192,
        seed: 23,
    };
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
