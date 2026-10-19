//! Freeze the live set into files the Python side reads.
//!
//! Training reads `train.jsonl` only. Eval reads `heldout.jsonl` and the
//! canary prompts. `data_sha256` names this exact dataset, and the adapter
//! manifest, the eval verdict and the ledger all carry it, so a promotion can
//! be traced back to the precise corrections it was trained on.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use serde::Serialize;

use crate::chain::sha256_hex;
use crate::store::{Correction, CorrectionStore};

#[derive(Debug, Serialize)]
pub struct ExportManifest {
    pub chain_head: String,
    pub n_train: usize,
    pub n_heldout: usize,
    pub n_heldout_gradable: usize,
    pub train_sha256: String,
    pub heldout_sha256: String,
    pub data_sha256: String,
    pub dir: String,
}

fn jsonl(rows: &[&Correction]) -> Result<String> {
    let mut s = String::new();
    for r in rows {
        s.push_str(&serde_json::to_string(r)?);
        s.push('\n');
    }
    Ok(s)
}

/// Writes `<out>/<chain_head[..12]>/{train,heldout}.jsonl` and `manifest.json`.
/// Rows are sorted by id so the same live set always produces the same bytes.
pub fn export(store: &CorrectionStore, out: &Path) -> Result<ExportManifest> {
    store.log().verify()?;
    let mut live = store.live()?;
    if live.is_empty() {
        bail!("no live corrections to export");
    }
    live.sort_by(|a, b| a.id.cmp(&b.id));
    let train: Vec<&Correction> = live.iter().filter(|c| c.split == "train").collect();
    let held: Vec<&Correction> = live.iter().filter(|c| c.split == "heldout").collect();
    let t = jsonl(&train)?;
    let h = jsonl(&held)?;
    let head = store.log().head()?;
    let dir: PathBuf = out.join(&head[..12]);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("train.jsonl"), &t)?;
    std::fs::write(dir.join("heldout.jsonl"), &h)?;
    let train_sha256 = sha256_hex(t.as_bytes());
    let heldout_sha256 = sha256_hex(h.as_bytes());
    let m = ExportManifest {
        chain_head: head,
        n_train: train.len(),
        n_heldout: held.len(),
        n_heldout_gradable: held.iter().filter(|c| c.check.is_gradable()).count(),
        data_sha256: sha256_hex(format!("{train_sha256}{heldout_sha256}").as_bytes()),
        train_sha256,
        heldout_sha256,
        dir: dir.display().to_string(),
    };
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&m)? + "\n",
    )?;
    Ok(m)
}
