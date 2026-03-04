// Shared text-ish token embedding and TF-IDF weighting helpers.
//
// Used by examples and tests that need a "less toy than random vectors" setup:
// char n-gram embeddings with TF-IDF-ish weights from two small documents.

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
