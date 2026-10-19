//! brain-adapt: Tier 2 learning for brain.
//!
//! Stage 1 (useful on its own): a correction store plus retrieval of past
//! corrections into context. Stage 2: export the store, train one LoRA
//! adapter from scratch (Python sidecar, `train/`), evaluate it against the
//! retrieval baseline. Stage 3: a hash-chained ledger that only promotes an
//! adapter whose verdict was produced under the locked PREREG, and rolls it
//! back automatically if live corrections get worse.
//!
//! The Rust loop links this crate for capture (`store`) and retrieval
//! (`retrieve`). See INTEGRATION.md.

pub mod chain;
pub mod embed;
pub mod export;
pub mod ledger;
pub mod retrieve;
pub mod store;

use std::path::{Path, PathBuf};

/// Default data folder, relative to the brain repo root.
pub const DEFAULT_DIR: &str = "adapt/data";

/// How retrieval runs. Read from the PREREG's `retrieval` block so the loop, training and
/// eval all retrieve the same way.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalConfig {
    pub k: usize,
    pub min_coverage: f64,
    /// Retrieval by meaning. None = word match only.
    pub semantic: Option<embed::SemanticConfig>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            k: 3,
            min_coverage: 0.5,
            semantic: None,
        }
    }
}

impl RetrievalConfig {
    /// From the PREREG JSON (`retrieval` block). Missing fields take defaults. The embedding URL
    /// can be overridden with the BRAIN_EMBED_URL environment variable.
    pub fn from_prereg_value(v: Option<&serde_json::Value>) -> Self {
        let r = v.and_then(|v| v.get("retrieval"));
        let get = |p: &str| r.and_then(|r| r.pointer(p));
        let semantic = get("/semantic")
            .filter(|s| !s.is_null())
            .map(|s| embed::SemanticConfig {
                url: std::env::var("BRAIN_EMBED_URL").ok().unwrap_or_else(|| {
                    s["url"]
                        .as_str()
                        .unwrap_or("http://localhost:1234/v1")
                        .to_string()
                }),
                model: s["model"]
                    .as_str()
                    .unwrap_or("text-embedding-nomic-embed-text-v1.5")
                    .to_string(),
                min_similarity: s["min_similarity"].as_f64().unwrap_or(0.7),
            });
        Self {
            k: get("/k").and_then(|x| x.as_u64()).unwrap_or(3) as usize,
            min_coverage: get("/min_coverage").and_then(|x| x.as_f64()).unwrap_or(0.5),
            semantic,
        }
    }

    pub fn from_prereg(path: &Path) -> Self {
        let v: Option<serde_json::Value> = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        Self::from_prereg_value(v.as_ref())
    }
}

/// What one retrieval did.
pub struct Retrieved {
    pub hits: Vec<retrieve::Hit>,
    pub explained: Vec<retrieve::Explained>,
    pub index_size: usize,
    /// "words" or "words+meaning".
    pub mode: &'static str,
    /// Set when meaning was configured but unavailable and retrieval fell back to words.
    pub warning: Option<String>,
}

/// Retrieve past corrections for `query` from the live train split.
/// `strict`: if retrieval by meaning is configured but the embedding server fails, return an
/// error instead of falling back to word match. Training and eval use strict mode so they never
/// silently retrieve differently from what the PREREG says; the interactive loop falls back.
pub fn retrieve(
    dir: &Path,
    query: &str,
    cfg: &RetrievalConfig,
    exclude: &[String],
    strict: bool,
) -> anyhow::Result<Retrieved> {
    let live = store::CorrectionStore::open(dir).live()?;
    let idx = retrieve::Index::build(&live);
    let mut warning = None;
    let mut sims: Option<Vec<f64>> = None;
    if let (Some(sc), false) = (&cfg.semantic, idx.is_empty()) {
        let attempt = (|| -> anyhow::Result<Vec<f64>> {
            let mut e = embed::Embedder::open(sc.clone(), dir)?;
            let docs = e.embed(&idx.doc_prompts(), embed::Kind::Document)?;
            let q = e.embed(&[query], embed::Kind::Query)?.remove(0);
            Ok(docs.iter().map(|d| embed::cosine(&q, d)).collect())
        })();
        match attempt {
            Ok(v) => sims = Some(v),
            Err(e) if strict => {
                return Err(e.context("retrieval by meaning is configured (strict mode)"))
            }
            Err(e) => {
                warning = Some(format!(
                    "retrieval by meaning unavailable, using word match only: {e:#}"
                ))
            }
        }
    }
    let min_sim = cfg
        .semantic
        .as_ref()
        .map(|s| s.min_similarity)
        .unwrap_or(1.0);
    let (hits, explained) = idx.search_full(
        query,
        cfg.k,
        cfg.min_coverage,
        exclude,
        sims.as_deref().map(|v| (v, min_sim)),
    );
    Ok(Retrieved {
        hits,
        explained,
        index_size: idx.len(),
        mode: if sims.is_some() {
            "words+meaning"
        } else {
            "words"
        },
        warning,
    })
}

/// Everything the loop needs at inference time, in one call: the context block
/// to put in front of the system prompt for this request (empty if nothing
/// relevant), and the adapter the sidecar should load (None = bare model).
pub struct Inference {
    pub context: String,
    pub hit_ids: Vec<String>,
    pub adapter: Option<ledger::Current>,
    pub mode: &'static str,
    pub warning: Option<String>,
}

pub fn for_request(dir: &Path, request: &str, cfg: &RetrievalConfig) -> anyhow::Result<Inference> {
    let r = retrieve(dir, request, cfg, &[], false)?;
    Ok(Inference {
        context: retrieve::format_context(&r.hits),
        hit_ids: r.hits.iter().map(|h| h.id.clone()).collect(),
        adapter: ledger::Ledger::open(dir).current()?,
        mode: r.mode,
        warning: r.warning,
    })
}

pub fn data_dir(arg: Option<PathBuf>) -> PathBuf {
    arg.unwrap_or_else(|| PathBuf::from(DEFAULT_DIR))
}
