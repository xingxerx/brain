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
pub mod export;
pub mod ledger;
pub mod retrieve;
pub mod store;

use std::path::{Path, PathBuf};

/// Default data folder, relative to the brain repo root.
pub const DEFAULT_DIR: &str = "adapt/data";

/// Everything the loop needs at inference time, in one call: the context block
/// to put in front of the system prompt for this request (empty if nothing
/// relevant), and the adapter the sidecar should load (None = bare model).
pub struct Inference {
    pub context: String,
    pub hit_ids: Vec<String>,
    pub adapter: Option<ledger::Current>,
}

pub fn for_request(
    dir: &Path,
    request: &str,
    k: usize,
    min_coverage: f64,
) -> anyhow::Result<Inference> {
    let store = store::CorrectionStore::open(dir);
    let idx = retrieve::Index::build(&store.live()?);
    let hits = idx.search(request, k, min_coverage);
    Ok(Inference {
        context: retrieve::format_context(&hits),
        hit_ids: hits.iter().map(|h| h.id.clone()).collect(),
        adapter: ledger::Ledger::open(dir).current()?,
    })
}

pub fn data_dir(arg: Option<PathBuf>) -> PathBuf {
    arg.unwrap_or_else(|| PathBuf::from(DEFAULT_DIR))
}
