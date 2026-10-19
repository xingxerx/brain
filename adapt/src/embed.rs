//! Retrieval by meaning: embeddings from an OpenAI-compatible `/embeddings` endpoint
//! (LM Studio serves `text-embedding-nomic-embed-text-v1.5` locally).
//!
//! Why: word-match retrieval only finds near-exact repeats. "where is the LIMEN repo" shares
//! one content word with "where is player-harness", so the player-harness correction never
//! reached it (run 13, 2026-10-18), even though it is the same lesson.
//!
//! Vectors are cached in `<data>/embeddings.jsonl`, keyed by model and exact input text, so each
//! correction is embedded once. The cache is derived data (gitignored); deleting it is safe.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::chain::sha256_hex;

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticConfig {
    /// Base URL of an OpenAI-compatible server, e.g. http://localhost:1234/v1
    pub url: String,
    pub model: String,
    /// Cosine similarity at or above which a correction counts as relevant.
    pub min_similarity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    Query,
    Document,
}

#[derive(Serialize, Deserialize)]
struct CacheLine {
    key: String,
    v: Vec<f32>,
}

pub struct Embedder {
    cfg: SemanticConfig,
    cache_path: PathBuf,
    cache: HashMap<String, Vec<f32>>,
    http: reqwest::blocking::Client,
}

/// nomic-embed models are trained with task prefixes; without them similarities are worse.
fn prefixed(model: &str, kind: Kind, text: &str) -> String {
    if model.to_lowercase().contains("nomic") {
        match kind {
            Kind::Query => format!("search_query: {text}"),
            Kind::Document => format!("search_document: {text}"),
        }
    } else {
        text.to_string()
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

impl Embedder {
    pub fn open(cfg: SemanticConfig, data_dir: &Path) -> Result<Self> {
        let cache_path = data_dir.join("embeddings.jsonl");
        let mut cache = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&cache_path) {
            // A torn last line (crash mid-write) is skipped, not fatal: it is only a cache.
            for line in text.lines() {
                if let Ok(c) = serde_json::from_str::<CacheLine>(line) {
                    cache.insert(c.key, c.v);
                }
            }
        }
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()?;
        Ok(Self {
            cfg,
            cache_path,
            cache,
            http,
        })
    }

    fn key(&self, input: &str) -> String {
        sha256_hex(format!("{}\0{}", self.cfg.model, input).as_bytes())
    }

    /// One vector per text, in order. Only uncached texts go to the server, in one request.
    pub fn embed(&mut self, texts: &[&str], kind: Kind) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<String> = texts
            .iter()
            .map(|t| prefixed(&self.cfg.model, kind, t))
            .collect();
        let missing: Vec<&String> = {
            let mut seen = std::collections::HashSet::new();
            inputs
                .iter()
                .filter(|i| !self.cache.contains_key(&self.key(i)) && seen.insert(i.as_str()))
                .collect()
        };
        if !missing.is_empty() {
            let url = format!("{}/embeddings", self.cfg.url.trim_end_matches('/'));
            let resp = self
                .http
                .post(&url)
                .json(&json!({"model": self.cfg.model, "input": missing}))
                .send()
                .with_context(|| format!("embedding server not reachable at {url}"))?;
            let status = resp.status();
            let body: serde_json::Value = resp
                .json()
                .with_context(|| format!("{url}: reply is not JSON"))?;
            if !status.is_success() {
                bail!("{url} returned {status}: {body}");
            }
            let data = body["data"]
                .as_array()
                .with_context(|| format!("{url}: no data array in reply"))?;
            if data.len() != missing.len() {
                bail!(
                    "{url}: asked for {} embeddings, got {}",
                    missing.len(),
                    data.len()
                );
            }
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.cache_path)?;
            for (i, d) in data.iter().enumerate() {
                // Servers may return items out of order; `index` says which input each one is.
                let idx = d["index"].as_u64().map(|x| x as usize).unwrap_or(i);
                let input = missing
                    .get(idx)
                    .with_context(|| format!("{url}: bad index {idx}"))?;
                let v: Vec<f32> = serde_json::from_value(d["embedding"].clone())
                    .with_context(|| format!("{url}: item {idx} has no embedding"))?;
                if v.is_empty() {
                    bail!("{url}: empty embedding");
                }
                let key = self.key(input);
                writeln!(
                    f,
                    "{}",
                    serde_json::to_string(&CacheLine {
                        key: key.clone(),
                        v: v.clone()
                    })?
                )?;
                self.cache.insert(key, v);
            }
            f.sync_all()?;
        }
        inputs
            .iter()
            .map(|i| {
                self.cache
                    .get(&self.key(i))
                    .cloned()
                    .context("embedding missing after fetch")
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn nomic_prefixes() {
        assert_eq!(
            prefixed("text-embedding-nomic-embed-text-v1.5", Kind::Query, "x"),
            "search_query: x"
        );
        assert_eq!(prefixed("other", Kind::Document, "x"), "x");
    }

    #[test]
    fn unreachable_server_is_an_error_not_a_panic() {
        let d = tempfile::tempdir().unwrap();
        let cfg = SemanticConfig {
            url: "http://127.0.0.1:9".into(),
            model: "m".into(),
            min_similarity: 0.7,
        };
        let mut e = Embedder::open(cfg, d.path()).unwrap();
        assert!(e.embed(&["hello"], Kind::Query).is_err());
    }
}
