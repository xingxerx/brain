//! Append-only, hash-chained JSONL log.
//!
//! Same posture as LIMEN's `RouterMemory` certificate ledger: every line is
//! `{"payload": {...}, "payload_sha256": "...", "chain": "..."}` where
//! `chain = sha256(prev_chain || payload_sha256)`. Editing or deleting any
//! earlier line breaks every chain value after it, so `verify()` catches it.
//! Deletion of a record is itself an appended event (a tombstone), never a
//! rewrite.
//!
//! Writes are fsynced before `append` returns (persist before ack). One writer
//! at a time is assumed; brain is a single local process.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// sha256 of bytes, lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Canonical JSON: serde_json's default `Map` is a BTreeMap, so keys come out
/// sorted and the encoding is stable across runs and machines.
pub fn canonical(v: &Value) -> String {
    serde_json::to_string(v).expect("Value always serializes")
}

pub fn chain_next(prev: &str, payload_sha: &str) -> String {
    sha256_hex(format!("{prev}{payload_sha}").as_bytes())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub payload: Value,
    pub payload_sha256: String,
    pub chain: String,
}

pub struct ChainedLog {
    path: PathBuf,
}

impl ChainedLog {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// All entries in file order. A missing file is an empty log.
    pub fn read_all(&self) -> Result<Vec<Entry>> {
        if !self.path.exists() {
            return Ok(vec![]);
        }
        let f = File::open(&self.path).with_context(|| format!("open {}", self.path.display()))?;
        let mut out = vec![];
        for (i, line) in BufReader::new(f).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let e: Entry = serde_json::from_str(&line).with_context(|| {
                format!(
                    "{} line {}: not a chained entry",
                    self.path.display(),
                    i + 1
                )
            })?;
            out.push(e);
        }
        Ok(out)
    }

    pub fn head(&self) -> Result<String> {
        Ok(self
            .read_all()?
            .last()
            .map(|e| e.chain.clone())
            .unwrap_or_else(|| GENESIS.to_string()))
    }

    /// Re-derive every hash. Returns the number of entries on success.
    pub fn verify(&self) -> Result<usize> {
        let entries = self.read_all()?;
        let mut prev = GENESIS.to_string();
        for (i, e) in entries.iter().enumerate() {
            let sha = sha256_hex(canonical(&e.payload).as_bytes());
            if sha != e.payload_sha256 {
                bail!(
                    "{} entry {}: payload hash mismatch (edited after write)",
                    self.path.display(),
                    i + 1
                );
            }
            let c = chain_next(&prev, &sha);
            if c != e.chain {
                bail!("{} entry {}: chain mismatch (an earlier entry was edited, removed or reordered)", self.path.display(), i + 1);
            }
            prev = c;
        }
        Ok(entries.len())
    }

    /// Append one payload. Refuses to write onto a log that does not verify.
    pub fn append(&self, payload: Value) -> Result<Entry> {
        self.verify()
            .context("refusing to append to a log that fails verification")?;
        let prev = self.head()?;
        let payload_sha256 = sha256_hex(canonical(&payload).as_bytes());
        let chain = chain_next(&prev, &payload_sha256);
        let entry = Entry {
            payload,
            payload_sha256,
            chain,
        };
        if let Some(dir) = self.path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let mut line = serde_json::to_string(&entry)?;
        line.push('\n');
        f.write_all(line.as_bytes())?;
        f.sync_all()?;
        Ok(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn append_and_verify() {
        let d = tempfile::tempdir().unwrap();
        let log = ChainedLog::new(d.path().join("x.jsonl"));
        assert_eq!(log.verify().unwrap(), 0);
        log.append(json!({"a": 1})).unwrap();
        log.append(json!({"b": 2})).unwrap();
        assert_eq!(log.verify().unwrap(), 2);
    }

    #[test]
    fn tamper_is_detected() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("x.jsonl");
        let log = ChainedLog::new(&p);
        log.append(json!({"a": 1})).unwrap();
        log.append(json!({"b": 2})).unwrap();
        let s = std::fs::read_to_string(&p)
            .unwrap()
            .replace("\"a\":1", "\"a\":9");
        std::fs::write(&p, s).unwrap();
        assert!(log.verify().is_err());
        assert!(log.append(json!({"c": 3})).is_err());
    }

    #[test]
    fn removed_line_is_detected() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("x.jsonl");
        let log = ChainedLog::new(&p);
        for i in 0..3 {
            log.append(json!({ "i": i })).unwrap();
        }
        let lines: Vec<String> = std::fs::read_to_string(&p)
            .unwrap()
            .lines()
            .map(String::from)
            .collect();
        std::fs::write(&p, format!("{}\n{}\n", lines[0], lines[2])).unwrap();
        assert!(log.verify().is_err());
    }

    #[test]
    fn canonical_is_key_sorted() {
        let v: Value = serde_json::from_str(r#"{"b":1,"a":2}"#).unwrap();
        assert_eq!(canonical(&v), r#"{"a":2,"b":1}"#);
    }
}
