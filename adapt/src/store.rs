//! The correction store: the actual asset of Tier 2.
//!
//! Adapters are disposable build outputs. This file is what you keep. Every
//! correction and every deletion is an event in a hash-chained log
//! (`corrections.jsonl`); the live set is the corrections minus the deleted
//! ones. Because adapters are always retrained from scratch on the live set,
//! deleting a correction really removes it from the next adapter.
//!
//! Rules enforced here (from the outside-view review):
//! - F5 leakage: the train/heldout split is fixed at capture time from the
//!   hash of the normalized prompt, and can never change afterwards. The same
//!   prompt always lands in the same split.
//! - F4 grading: every correction declares a deterministic check (decision
//!   label, required substrings, or exact match) so eval never needs a model
//!   to grade it. `none` is allowed but those rows are reported as ungradable.
//! - F8 bad data: a new correction whose prompt nearly matches a live one but
//!   whose correction differs is refused unless you supersede the old one or
//!   explicitly allow the conflict. Only local sources are accepted.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::chain::{sha256_hex, ChainedLog};

pub const DECISIONS: [&str; 4] = ["act", "answer", "ask", "abort"];
pub const KINDS: [&str; 4] = ["decision", "answer", "format", "other"];
/// Who may write corrections. Tool output, web pages and model text are never
/// sources: anything that can put text in front of the model must not be able
/// to teach it.
pub const SOURCES: [&str; 2] = ["local-cli", "brain-user"];
/// Near-duplicate threshold on prompt token sets (Jaccard).
pub const CONFLICT_JACCARD: f64 = 0.8;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Check {
    /// The first decision the model emits must equal `value`.
    Decision { value: String },
    /// Every string in `value` must appear in the output (case-insensitive).
    Contains { value: Vec<String> },
    /// The output, whitespace-normalized, must equal the correction.
    Exact,
    /// No automatic check. Excluded from gate scoring, listed as ungradable.
    None,
}

impl Check {
    pub fn is_gradable(&self) -> bool {
        !matches!(self, Check::None)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Correction {
    pub id: String,
    pub ts: u64,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub model_output: String,
    pub correction: String,
    pub kind: String,
    pub check: Check,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    pub source: String,
    pub split: String,
    pub prompt_hash: String,
}

/// What a caller supplies. Everything else is derived.
#[derive(Debug, Clone, Default)]
pub struct NewCorrection {
    pub prompt: String,
    pub system: Option<String>,
    pub model_output: String,
    pub correction: String,
    pub kind: String,
    pub check: Option<Check>,
    pub tag: Option<String>,
    pub source: String,
}

#[derive(Debug, Default, Clone)]
pub struct AddOptions {
    pub supersede: Vec<String>,
    pub allow_conflict: bool,
    /// Fraction of prompts (by hash) that go to heldout. Fixed per store.
    pub heldout_frac: f64,
    /// Seconds since epoch; tests pass a fixed value.
    pub now: Option<u64>,
}

#[derive(Debug)]
pub struct Conflict {
    pub with: Vec<String>,
}

impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "conflicts with live correction(s) {} (near-identical prompt, different correction). \
             Pass --supersede <id> to replace, or --allow-conflict if both are right",
            self.with.join(", ")
        )
    }
}
impl std::error::Error for Conflict {}

/// The first decision in a model output. Same rule as `train/common.py`:
/// the first `"decision": "<word>"` in the text, else the whole trimmed text
/// if it is a bare decision word. Anything else is None.
pub fn extract_decision(text: &str) -> Option<String> {
    let key = "\"decision\"";
    let mut rest = text;
    while let Some(i) = rest.find(key) {
        let after = rest[i + key.len()..].trim_start();
        if let Some(after) = after.strip_prefix(':') {
            let after = after.trim_start();
            if let Some(after) = after.strip_prefix('"') {
                let word: String = after
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if after[word.len()..].starts_with('"') && DECISIONS.contains(&word.as_str()) {
                    return Some(word);
                }
            }
        }
        rest = &rest[i + key.len()..];
    }
    let t = text.trim().to_lowercase();
    DECISIONS.contains(&t.as_str()).then_some(t)
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Lowercase, keep letters and digits, collapse everything else to single spaces.
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = true;
    for c in s.chars().flat_map(|c| c.to_lowercase()) {
        if c.is_alphanumeric() {
            out.push(c);
            space = false;
        } else if !space {
            out.push(' ');
            space = true;
        }
    }
    out.trim_end().to_string()
}

pub fn tokens(s: &str) -> Vec<String> {
    normalize(s)
        .split(' ')
        .filter(|t| !t.is_empty())
        .map(String::from)
        .collect()
}

fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f64;
    let uni = a.union(b).count() as f64;
    inter / uni
}

/// Split from the prompt hash: first two bytes as a fraction in [0, 1).
pub fn split_for(prompt_hash: &str, heldout_frac: f64) -> &'static str {
    let v = u16::from_str_radix(&prompt_hash[..4], 16).unwrap_or(0) as f64 / 65536.0;
    if v < heldout_frac {
        "heldout"
    } else {
        "train"
    }
}

pub struct CorrectionStore {
    log: ChainedLog,
}

impl CorrectionStore {
    pub fn open(dir: impl AsRef<Path>) -> Self {
        Self {
            log: ChainedLog::new(dir.as_ref().join("corrections.jsonl")),
        }
    }

    pub fn log(&self) -> &ChainedLog {
        &self.log
    }

    /// Live corrections, in insertion order.
    pub fn live(&self) -> Result<Vec<Correction>> {
        let mut by_id: BTreeMap<String, (usize, Correction)> = BTreeMap::new();
        for (i, e) in self.log.read_all()?.into_iter().enumerate() {
            match e.payload.get("event").and_then(Value::as_str) {
                Some("correction") => {
                    let c: Correction = serde_json::from_value(e.payload["correction"].clone())?;
                    by_id.insert(c.id.clone(), (i, c));
                }
                Some("delete") => {
                    if let Some(id) = e.payload.get("id").and_then(Value::as_str) {
                        by_id.remove(id);
                    }
                }
                other => bail!("unknown event {:?} in {}", other, self.log.path().display()),
            }
        }
        let mut v: Vec<(usize, Correction)> = by_id.into_values().collect();
        v.sort_by_key(|(i, _)| *i);
        Ok(v.into_iter().map(|(_, c)| c).collect())
    }

    pub fn get(&self, id: &str) -> Result<Option<Correction>> {
        Ok(self.live()?.into_iter().find(|c| c.id == id))
    }

    /// The split this store uses. The first correction pins it: changing
    /// heldout_frac later would move prompts between splits, so it is refused.
    fn pinned_frac(&self) -> Result<Option<f64>> {
        for e in self.log.read_all()? {
            if let Some(f) = e.payload.get("heldout_frac").and_then(Value::as_f64) {
                return Ok(Some(f));
            }
        }
        Ok(None)
    }

    pub fn add(&self, new: NewCorrection, opts: &AddOptions) -> Result<Correction> {
        if !SOURCES.contains(&new.source.as_str()) {
            bail!("source {:?} is not allowed; only {:?}", new.source, SOURCES);
        }
        if !KINDS.contains(&new.kind.as_str()) {
            bail!("kind must be one of {:?}", KINDS);
        }
        if new.prompt.trim().is_empty() || new.correction.trim().is_empty() {
            bail!("prompt and correction must be non-empty");
        }
        let check = new.check.clone().unwrap_or(Check::None);
        match (&new.kind[..], &check) {
            ("decision", Check::Decision { value }) => {
                if !DECISIONS.contains(&value.as_str()) {
                    bail!("decision check must be one of {:?}", DECISIONS);
                }
            }
            ("decision", _) => bail!(
                "kind=decision needs a decision check (--check-decision act|answer|ask|abort)"
            ),
            (_, Check::Decision { value }) if !DECISIONS.contains(&value.as_str()) => {
                bail!("decision check must be one of {:?}", DECISIONS)
            }
            (_, Check::Contains { value }) if value.iter().all(|s| s.trim().is_empty()) => {
                bail!("contains check needs at least one non-empty string")
            }
            _ => {}
        }
        if let Check::Decision { value } = &check {
            if extract_decision(&new.correction).as_deref() != Some(value.as_str()) {
                bail!("decision check {:?} does not match the decision in your correction text ({:?}); the correction would fail its own check",
                    value, extract_decision(&new.correction));
            }
        }
        if let Check::Exact = &check {
            if normalize(&new.correction).is_empty() {
                bail!("exact check needs a correction with some letters or digits");
            }
        }
        if let Check::Contains { value } = &check {
            let corr = new.correction.to_lowercase();
            for s in value {
                if !corr.contains(&s.to_lowercase()) {
                    bail!("contains check {:?} does not appear in your own correction, so the correction would fail its own check", s);
                }
            }
        }

        let frac = match self.pinned_frac()? {
            Some(f) => {
                if (f - opts.heldout_frac).abs() > 1e-9 {
                    bail!("this store's heldout_frac is pinned at {f}; refusing {} (it would move prompts between splits)", opts.heldout_frac);
                }
                f
            }
            None => {
                if !(0.05..=0.5).contains(&opts.heldout_frac) {
                    bail!("heldout_frac must be between 0.05 and 0.5");
                }
                opts.heldout_frac
            }
        };

        let norm_prompt = normalize(&new.prompt);
        let prompt_hash = sha256_hex(norm_prompt.as_bytes());
        let norm_corr = normalize(&new.correction);
        let id = format!(
            "c_{}",
            &sha256_hex(format!("{prompt_hash}|{norm_corr}").as_bytes())[..12]
        );

        let live = self.live()?;
        if live.iter().any(|c| c.id == id) {
            bail!("duplicate: correction {id} already exists with the same prompt and correction");
        }
        for s in &opts.supersede {
            if !live.iter().any(|c| &c.id == s) {
                bail!("--supersede {s}: no live correction with that id");
            }
        }
        let new_toks: BTreeSet<String> = tokens(&new.prompt).into_iter().collect();
        let conflicts: Vec<String> = live
            .iter()
            .filter(|c| !opts.supersede.contains(&c.id))
            .filter(|c| normalize(&c.correction) != norm_corr)
            .filter(|c| {
                let t: BTreeSet<String> = tokens(&c.prompt).into_iter().collect();
                jaccard(&new_toks, &t) >= CONFLICT_JACCARD
            })
            .map(|c| c.id.clone())
            .collect();
        if !conflicts.is_empty() && !opts.allow_conflict {
            return Err(Conflict { with: conflicts }.into());
        }

        let ts = opts.now.unwrap_or_else(now_secs);
        for s in &opts.supersede {
            self.log.append(json!({"event": "delete", "id": s, "ts": ts, "reason": format!("superseded by {id}")}))?;
        }
        let c = Correction {
            id,
            ts,
            prompt: new.prompt,
            system: new.system,
            model_output: new.model_output,
            correction: new.correction,
            kind: new.kind,
            check,
            tag: new.tag,
            source: new.source,
            split: split_for(&prompt_hash, frac).to_string(),
            prompt_hash,
        };
        self.log.append(json!({
            "event": "correction",
            "heldout_frac": frac,
            "conflicts_allowed": if opts.allow_conflict { conflicts } else { vec![] },
            "correction": serde_json::to_value(&c)?,
        }))?;
        Ok(c)
    }

    pub fn delete(&self, id: &str, reason: &str) -> Result<()> {
        if self.get(id)?.is_none() {
            bail!("no live correction {id}");
        }
        self.log
            .append(json!({"event": "delete", "id": id, "ts": now_secs(), "reason": reason}))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nc(prompt: &str, corr: &str) -> NewCorrection {
        NewCorrection {
            prompt: prompt.into(),
            model_output: "{\"decision\":\"act\"}".into(),
            correction: corr.into(),
            kind: "decision".into(),
            check: Some(Check::Decision {
                value: "ask".into(),
            }),
            source: "local-cli".into(),
            ..Default::default()
        }
    }

    fn opts() -> AddOptions {
        AddOptions {
            heldout_frac: 0.2,
            now: Some(1),
            ..Default::default()
        }
    }

    #[test]
    fn split_is_fixed_by_prompt() {
        let d = tempfile::tempdir().unwrap();
        let s = CorrectionStore::open(d.path());
        let a = s
            .add(
                nc("Delete the file please", "{\"decision\":\"ask\"}"),
                &opts(),
            )
            .unwrap();
        s.delete(&a.id, "test").unwrap();
        let b = s
            .add(
                nc("delete   the FILE, please!", "{\"decision\":\"ask\"}"),
                &opts(),
            )
            .unwrap();
        assert_eq!(a.split, b.split);
        assert_eq!(a.prompt_hash, b.prompt_hash);
    }

    #[test]
    fn heldout_fraction_is_roughly_right_and_pinned() {
        let d = tempfile::tempdir().unwrap();
        let s = CorrectionStore::open(d.path());
        for i in 0..400 {
            s.add(
                nc(
                    &format!("task number {i} about thing {}", i * 7919),
                    "{\"decision\":\"ask\"}",
                ),
                &AddOptions {
                    allow_conflict: true,
                    ..opts()
                },
            )
            .unwrap();
        }
        let held = s
            .live()
            .unwrap()
            .iter()
            .filter(|c| c.split == "heldout")
            .count();
        assert!((50..=110).contains(&held), "heldout {held}/400");
        let bad = AddOptions {
            heldout_frac: 0.3,
            ..opts()
        };
        assert!(s.add(nc("another one", "x"), &bad).is_err());
    }

    #[test]
    fn conflict_detected_and_supersede_works() {
        let d = tempfile::tempdir().unwrap();
        let s = CorrectionStore::open(d.path());
        let a = s
            .add(
                nc(
                    "what is the total value of all stock in inventory.csv",
                    "{\"decision\":\"ask\"}",
                ),
                &opts(),
            )
            .unwrap();
        let mut b = nc(
            "What is the total value of all the stock in inventory.csv?",
            "{\"decision\":\"act\"}",
        );
        b.check = Some(Check::Decision {
            value: "act".into(),
        });
        let err = s.add(b.clone(), &opts()).unwrap_err();
        assert!(err.downcast_ref::<Conflict>().is_some());
        let o = AddOptions {
            supersede: vec![a.id.clone()],
            ..opts()
        };
        let c = s.add(b, &o).unwrap();
        let live = s.live().unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].id, c.id);
        assert_eq!(s.log().verify().unwrap(), 3);
    }

    #[test]
    fn duplicate_refused() {
        let d = tempfile::tempdir().unwrap();
        let s = CorrectionStore::open(d.path());
        s.add(nc("p one two", "{\"decision\":\"ask\"}"), &opts())
            .unwrap();
        assert!(s
            .add(nc("P one two", "{\"decision\":\"ask\"}"), &opts())
            .is_err());
    }

    #[test]
    fn untrusted_source_and_bad_checks_refused() {
        let d = tempfile::tempdir().unwrap();
        let s = CorrectionStore::open(d.path());
        let mut x = nc("p", "c");
        x.source = "tool-output".into();
        assert!(s.add(x, &opts()).is_err());
        let mut y = nc("p", "c");
        y.check = None;
        assert!(
            s.add(y, &opts()).is_err(),
            "decision kind needs a decision check"
        );
        let mut z = nc("p", "the answer is 42");
        z.kind = "answer".into();
        z.check = Some(Check::Contains {
            value: vec!["43".into()],
        });
        assert!(
            s.add(z, &opts()).is_err(),
            "check must pass on the correction itself"
        );
    }

    #[test]
    fn decision_extraction() {
        assert_eq!(
            extract_decision(r#"{"decision": "ask", "question": "which file?"}"#).as_deref(),
            Some("ask")
        );
        assert_eq!(
            extract_decision(r#"{"thought":"x","decision":"act","tool":"calculator"}"#).as_deref(),
            Some("act")
        );
        assert_eq!(extract_decision(" Abort "), Some("abort".into()));
        assert_eq!(extract_decision(r#"{"decision": "maybe"}"#), None);
        assert_eq!(extract_decision("I would ask"), None);
        let d = tempfile::tempdir().unwrap();
        let s = CorrectionStore::open(d.path());
        assert!(
            s.add(nc("q", r#"{"decision":"act"}"#), &opts()).is_err(),
            "check ask vs text act"
        );
    }

    #[test]
    fn delete_removes_from_live() {
        let d = tempfile::tempdir().unwrap();
        let s = CorrectionStore::open(d.path());
        let a = s
            .add(nc("alpha", "{\"decision\":\"ask\"}"), &opts())
            .unwrap();
        s.add(nc("beta gamma", "{\"decision\":\"ask\"}"), &opts())
            .unwrap();
        s.delete(&a.id, "wrong").unwrap();
        assert_eq!(s.live().unwrap().len(), 1);
        assert!(s.delete(&a.id, "again").is_err());
    }
}

#[cfg(test)]
mod parity {
    //! The same cases are run by train/test_common.py against the Python side.
    use super::*;

    #[test]
    fn matches_shared_cases() {
        let v: Value = serde_json::from_str(include_str!("../tests/parity_cases.json")).unwrap();
        for c in v["normalize"].as_array().unwrap() {
            assert_eq!(
                normalize(c[0].as_str().unwrap()),
                c[1].as_str().unwrap(),
                "normalize {:?}",
                c[0]
            );
        }
        for c in v["extract_decision"].as_array().unwrap() {
            assert_eq!(
                extract_decision(c[0].as_str().unwrap()).as_deref(),
                c[1].as_str(),
                "extract {:?}",
                c[0]
            );
        }
    }
}
