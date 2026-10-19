//! Adapter registry and promotion gate (F11, F7, stage 3).
//!
//! Hash-chained like the correction store (`adapters.jsonl`). The gate lives
//! here, in Rust, not in the Python that produced the numbers: Python writes a
//! verdict file, this module decides whether that verdict is allowed to change
//! what the loop serves.
//!
//! An adapter can be promoted only if:
//! - the PREREG was locked before the eval, and has not changed since;
//! - the verdict was produced under that locked PREREG;
//! - the verdict is for this adapter, its base model hash, its dataset and the
//!   serving precision the PREREG names;
//! - it is not a smoke run, and the verdict says pass.
//!
//! Promotion is provisional. `check_live` watches corrections per 100
//! interactions after promotion and rolls back automatically if the rate gets
//! worse than the baseline by more than the PREREG tolerance.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};

use crate::chain::{sha256_hex, ChainedLog};
use crate::store::now_secs;

pub struct Ledger {
    log: ChainedLog,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Current {
    pub adapter_id: String,
    pub adapter_dir: String,
    pub base_hash: String,
    pub data_sha256: String,
    pub confirmed: bool,
    pub baseline_rate: f64,
    pub promoted_ts: u64,
}

#[derive(Debug, PartialEq)]
pub enum LiveOutcome {
    KeepObserving { needed: u64 },
    Confirmed,
    RolledBack { rate: f64, limit: f64 },
}

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(String::from)
}

/// Any string value "TODO" anywhere in the PREREG means it is not ready to lock.
fn find_todo(v: &Value, path: &str, out: &mut Vec<String>) {
    match v {
        Value::String(x) if x.trim().eq_ignore_ascii_case("todo") => out.push(path.to_string()),
        Value::Object(m) => m
            .iter()
            .for_each(|(k, x)| find_todo(x, &format!("{path}.{k}"), out)),
        Value::Array(a) => a
            .iter()
            .enumerate()
            .for_each(|(i, x)| find_todo(x, &format!("{path}[{i}]"), out)),
        _ => {}
    }
}

pub fn file_sha(p: &Path) -> Result<String> {
    Ok(sha256_hex(
        &std::fs::read(p).with_context(|| format!("read {}", p.display()))?,
    ))
}

fn read_json(p: &Path) -> Result<Value> {
    serde_json::from_str(
        &std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?,
    )
    .with_context(|| format!("{} is not JSON", p.display()))
}

impl Ledger {
    pub fn open(dir: impl AsRef<Path>) -> Self {
        Self {
            log: ChainedLog::new(dir.as_ref().join("adapters.jsonl")),
        }
    }

    pub fn log(&self) -> &ChainedLog {
        &self.log
    }

    fn events(&self) -> Result<Vec<Value>> {
        Ok(self
            .log
            .read_all()?
            .into_iter()
            .map(|e| e.payload)
            .collect())
    }

    pub fn locked_prereg(&self) -> Result<Option<String>> {
        Ok(self
            .events()?
            .iter()
            .rev()
            .find(|e| s(e, "event").as_deref() == Some("prereg_lock"))
            .and_then(|e| s(e, "prereg_sha256")))
    }

    pub fn lock_prereg(&self, prereg: &Path) -> Result<String> {
        let v = read_json(prereg)?;
        let mut todos = vec![];
        find_todo(&v, "$", &mut todos);
        if !todos.is_empty() {
            bail!(
                "PREREG still has TODO at {}; fill them in before locking",
                todos.join(", ")
            );
        }
        let sha = file_sha(prereg)?;
        if self.locked_prereg()?.as_deref() == Some(sha.as_str()) {
            return Ok(sha);
        }
        self.log
            .append(json!({"event": "prereg_lock", "prereg_sha256": sha, "ts": now_secs()}))?;
        Ok(sha)
    }

    fn registration(&self, adapter_id: &str) -> Result<Option<Value>> {
        Ok(self.events()?.into_iter().find(|e| {
            s(e, "event").as_deref() == Some("register")
                && s(e, "adapter_id").as_deref() == Some(adapter_id)
        }))
    }

    /// Reads `<adapter_dir>/manifest.json` written by train_lora.py.
    pub fn register(&self, adapter_dir: &Path) -> Result<String> {
        let m = read_json(&adapter_dir.join("manifest.json"))?;
        let id = s(&m, "adapter_id").context("manifest has no adapter_id")?;
        for k in ["base_hash", "data_sha256", "prereg_sha256"] {
            if s(&m, k).is_none() {
                bail!("manifest has no {k}");
            }
        }
        if self.registration(&id)?.is_some() {
            bail!("adapter {id} is already registered");
        }
        self.log.append(json!({
            "event": "register", "adapter_id": id, "adapter_dir": adapter_dir.display().to_string(),
            "manifest": m, "ts": now_secs(),
        }))?;
        Ok(id)
    }

    /// The promotion gate. `prereg` is the PREREG file as it is on disk now.
    pub fn promote(
        &self,
        adapter_id: &str,
        verdict_path: &Path,
        prereg: &Path,
        baseline_rate: f64,
        reviewed: bool,
    ) -> Result<()> {
        let reg = self
            .registration(adapter_id)?
            .with_context(|| format!("adapter {adapter_id} is not registered"))?;
        let man = &reg["manifest"];
        let locked = self.locked_prereg()?.context(
            "no PREREG lock in the ledger; run `brain-adapt prereg lock` before any eval",
        )?;
        let now_sha = file_sha(prereg)?;
        if now_sha != locked {
            bail!("PREREG changed since it was locked ({} now vs {} locked). Re-lock and re-run the eval", &now_sha[..12], &locked[..12]);
        }
        let pr = read_json(prereg)?;
        let v = read_json(verdict_path)?;
        let mut why = vec![];
        let eq = |k: &str, want: Option<String>, why: &mut Vec<String>| {
            if s(&v, k) != want {
                why.push(format!("verdict {k}={:?}, expected {:?}", s(&v, k), want));
            }
        };
        eq("adapter_id", Some(adapter_id.to_string()), &mut why);
        eq("prereg_sha256", Some(locked.clone()), &mut why);
        eq("base_hash", s(man, "base_hash"), &mut why);
        eq("data_sha256", s(man, "data_sha256"), &mut why);
        eq("precision", s(&pr, "serving_precision"), &mut why);
        eq("thinking", s(&pr, "thinking"), &mut why);
        if s(man, "prereg_sha256").as_deref() != Some(locked.as_str()) {
            why.push("adapter was trained under a different PREREG than the locked one".into());
        }
        if v.get("smoke").and_then(Value::as_bool) != Some(false)
            || man.get("smoke").and_then(Value::as_bool) != Some(false)
        {
            why.push("smoke runs can never be promoted".into());
        }
        if s(&v, "verdict").as_deref() != Some("pass") {
            why.push(format!(
                "verdict is {:?}: {}",
                s(&v, "verdict"),
                v.get("reasons").map(|r| r.to_string()).unwrap_or_default()
            ));
        }
        if !(baseline_rate.is_finite() && baseline_rate >= 0.0) {
            why.push("baseline rate must be a number >= 0 (corrections per 100 interactions before this adapter)".into());
        }
        if !reviewed {
            why.push(format!(
                "human review not confirmed: read {} and pass --reviewed",
                s(&v, "review_file").unwrap_or_else(|| "the review file".into())
            ));
        }
        if self.current()?.map(|c| c.adapter_id).as_deref() == Some(adapter_id) {
            why.push("already the current adapter".into());
        }
        if !why.is_empty() {
            bail!("promotion refused:\n  - {}", why.join("\n  - "));
        }
        self.log.append(json!({
            "event": "promote", "adapter_id": adapter_id, "eval_sha256": file_sha(verdict_path)?,
            "baseline_rate": baseline_rate, "reviewed": true, "ts": now_secs(),
        }))?;
        Ok(())
    }

    /// Replays promote / confirm / rollback into a stack. The top is current.
    pub fn current(&self) -> Result<Option<Current>> {
        let ev = self.events()?;
        let mut stack: Vec<Current> = vec![];
        for e in &ev {
            match s(e, "event").as_deref() {
                Some("promote") => {
                    let id = s(e, "adapter_id").unwrap_or_default();
                    let reg = ev.iter().find(|r| {
                        s(r, "event").as_deref() == Some("register")
                            && s(r, "adapter_id").as_deref() == Some(id.as_str())
                    });
                    let (dir, man) = reg
                        .map(|r| {
                            (
                                s(r, "adapter_dir").unwrap_or_default(),
                                r["manifest"].clone(),
                            )
                        })
                        .unwrap_or_default();
                    stack.retain(|c| c.adapter_id != id);
                    stack.push(Current {
                        adapter_id: id,
                        adapter_dir: dir,
                        base_hash: s(&man, "base_hash").unwrap_or_default(),
                        data_sha256: s(&man, "data_sha256").unwrap_or_default(),
                        confirmed: false,
                        baseline_rate: e
                            .get("baseline_rate")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0),
                        promoted_ts: e.get("ts").and_then(Value::as_u64).unwrap_or(0),
                    });
                }
                Some("confirm") => {
                    if let Some(top) = stack.last_mut() {
                        if Some(top.adapter_id.clone()) == s(e, "adapter_id") {
                            top.confirmed = true;
                        }
                    }
                }
                Some("rollback")
                    if stack.last().map(|c| c.adapter_id.clone()) == s(e, "adapter_id") =>
                {
                    stack.pop();
                }
                _ => {}
            }
        }
        Ok(stack.pop())
    }

    pub fn rollback(&self, reason: &str, auto: bool) -> Result<Option<Current>> {
        let cur = self
            .current()?
            .context("nothing is promoted; the loop is already on the bare model")?;
        self.log.append(json!({"event": "rollback", "adapter_id": cur.adapter_id, "reason": reason, "auto": auto, "ts": now_secs()}))?;
        self.current()
    }

    /// Corrections and interactions counted since the current promotion.
    pub fn check_live(
        &self,
        interactions: u64,
        corrections: u64,
        prereg: &Path,
    ) -> Result<LiveOutcome> {
        let cur = self.current()?.context("nothing is promoted")?;
        if cur.confirmed {
            bail!("adapter {} is already confirmed", cur.adapter_id);
        }
        if corrections > interactions {
            bail!("more corrections than interactions");
        }
        let pr = read_json(prereg)?;
        let min = pr
            .pointer("/live/min_interactions")
            .and_then(Value::as_u64)
            .context("PREREG live.min_interactions missing")?;
        let tol = pr
            .pointer("/live/tolerance")
            .and_then(Value::as_f64)
            .context("PREREG live.tolerance missing")?;
        let rate = if interactions == 0 {
            0.0
        } else {
            100.0 * corrections as f64 / interactions as f64
        };
        // Floor of 1 per 100 so a zero baseline does not roll back on the first correction.
        let limit = cur.baseline_rate.max(1.0) * (1.0 + tol);
        if interactions >= min.min(20) && rate > limit && corrections >= 3 {
            self.log.append(json!({
                "event": "rollback", "adapter_id": cur.adapter_id, "auto": true, "ts": now_secs(),
                "reason": format!("live correction rate {rate:.2}/100 over limit {limit:.2}/100 after {interactions} interactions"),
            }))?;
            return Ok(LiveOutcome::RolledBack { rate, limit });
        }
        if interactions < min {
            return Ok(LiveOutcome::KeepObserving {
                needed: min - interactions,
            });
        }
        self.log.append(json!({"event": "confirm", "adapter_id": cur.adapter_id, "rate": rate, "interactions": interactions, "ts": now_secs()}))?;
        Ok(LiveOutcome::Confirmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct Fx {
        _d: tempfile::TempDir,
        dir: std::path::PathBuf,
        prereg: std::path::PathBuf,
    }

    fn fx() -> Fx {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().to_path_buf();
        let prereg = dir.join("prereg.json");
        fs::write(&prereg, r#"{"serving_precision":"int8","thinking":"off","live":{"min_interactions":100,"tolerance":0.25}}"#).unwrap();
        Fx { _d: d, dir, prereg }
    }

    fn adapter(f: &Fx, id: &str, locked: &str, smoke: bool) -> std::path::PathBuf {
        let a = f.dir.join(id);
        fs::create_dir_all(&a).unwrap();
        fs::write(a.join("manifest.json"), json!({"adapter_id": id, "base_hash": "B", "data_sha256": "D", "prereg_sha256": locked, "smoke": smoke}).to_string()).unwrap();
        a
    }

    fn verdict(f: &Fx, id: &str, locked: &str, pass: bool) -> std::path::PathBuf {
        let p = f.dir.join(format!("verdict_{id}.json"));
        fs::write(&p, json!({"adapter_id": id, "prereg_sha256": locked, "base_hash": "B", "data_sha256": "D",
            "precision": "int8", "thinking": "off", "smoke": false, "verdict": if pass {"pass"} else {"fail"}, "reasons": []}).to_string()).unwrap();
        p
    }

    #[test]
    fn todo_blocks_lock() {
        let f = fx();
        fs::write(&f.prereg, r#"{"serving_precision":"TODO"}"#).unwrap();
        assert!(Ledger::open(&f.dir).lock_prereg(&f.prereg).is_err());
    }

    #[test]
    fn full_promote_rollback_cycle() {
        let f = fx();
        let l = Ledger::open(&f.dir);
        let locked = l.lock_prereg(&f.prereg).unwrap();
        let a1 = adapter(&f, "a_one", &locked, false);
        let a2 = adapter(&f, "a_two", &locked, false);
        l.register(&a1).unwrap();
        l.register(&a2).unwrap();
        assert!(l.register(&a1).is_err());
        l.promote(
            "a_one",
            &verdict(&f, "a_one", &locked, true),
            &f.prereg,
            5.0,
            true,
        )
        .unwrap();
        assert_eq!(l.current().unwrap().unwrap().adapter_id, "a_one");
        assert_eq!(
            l.check_live(150, 5, &f.prereg).unwrap(),
            LiveOutcome::Confirmed
        );
        l.promote(
            "a_two",
            &verdict(&f, "a_two", &locked, true),
            &f.prereg,
            3.0,
            true,
        )
        .unwrap();
        // 10 per 100 is worse than 3 * 1.25: automatic rollback to a_one.
        assert!(matches!(
            l.check_live(50, 5, &f.prereg).unwrap(),
            LiveOutcome::RolledBack { .. }
        ));
        let cur = l.current().unwrap().unwrap();
        assert_eq!(cur.adapter_id, "a_one");
        assert!(cur.confirmed);
        l.rollback("manual", false).unwrap();
        assert!(l.current().unwrap().is_none());
        assert_eq!(l.log().verify().unwrap(), 8);
    }

    #[test]
    fn gate_refusals() {
        let f = fx();
        let l = Ledger::open(&f.dir);
        // No lock yet.
        let a = adapter(&f, "a_x", "L", false);
        l.register(&a).unwrap();
        assert!(l
            .promote("a_x", &verdict(&f, "a_x", "L", true), &f.prereg, 1.0, true)
            .is_err());
        let locked = l.lock_prereg(&f.prereg).unwrap();
        // Adapter trained under a different PREREG.
        assert!(l
            .promote(
                "a_x",
                &verdict(&f, "a_x", &locked, true),
                &f.prereg,
                1.0,
                true
            )
            .is_err());
        // Failing verdict.
        let b = adapter(&f, "a_y", &locked, false);
        l.register(&b).unwrap();
        assert!(l
            .promote(
                "a_y",
                &verdict(&f, "a_y", &locked, false),
                &f.prereg,
                1.0,
                true
            )
            .is_err());
        // Smoke.
        let c = adapter(&f, "a_z", &locked, true);
        l.register(&c).unwrap();
        assert!(l
            .promote(
                "a_z",
                &verdict(&f, "a_z", &locked, true),
                &f.prereg,
                1.0,
                true
            )
            .is_err());
        // Not reviewed.
        let r = adapter(&f, "a_r", &locked, false);
        l.register(&r).unwrap();
        assert!(l
            .promote(
                "a_r",
                &verdict(&f, "a_r", &locked, true),
                &f.prereg,
                1.0,
                false
            )
            .is_err());
        // PREREG edited after lock.
        fs::write(&f.prereg, r#"{"serving_precision":"bf16","thinking":"off","live":{"min_interactions":1,"tolerance":9}}"#).unwrap();
        assert!(l
            .promote(
                "a_y",
                &verdict(&f, "a_y", &locked, true),
                &f.prereg,
                1.0,
                true
            )
            .is_err());
        assert!(l.current().unwrap().is_none());
    }
}
