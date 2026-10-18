//! Append-only JSONL run log. One line per step, written as it happens,
//! so a crash mid-run still leaves everything up to that point.

use crate::decision::{GateOutput, Proposal};
use crate::llm::ChatMessage;
use anyhow::Result;
use serde::Serialize;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[derive(Debug, Serialize)]
pub struct StepRecord<'a> {
    pub ts_ms: u128,
    pub run_id: &'a str,
    pub task_id: &'a str,
    pub step: usize,
    pub latency_ms: u128,
    /// Generated tokens (thinking included), when the server reports usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    pub reasoning_chars: usize,
    /// The model's raw reply text.
    pub raw: &'a str,
    pub proposal: Option<&'a Proposal>,
    pub parse_error: Option<&'a str>,
    pub gate: Option<&'a GateOutput>,
    /// Shadow gate: logged for comparison, never acted on.
    pub shadow: Option<&'a GateOutput>,
    pub shadow_error: Option<&'a str>,
    /// When the gate overrode the model and the model's reply had no payload for the
    /// gated decision, the loop asks once for it. That reply (not gated again) lands here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refill_raw: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refill_latency_ms: Option<u128>,
    pub observation: Option<&'a str>,
    /// Full conversation state the model saw on this step, so any step can be replayed.
    pub messages: &'a [ChatMessage],
}

pub struct RunLog {
    pub run_id: String,
    pub dir: PathBuf,
    steps: BufWriter<File>,
}

impl RunLog {
    pub fn create(runs_dir: &Path, label: &str) -> Result<Self> {
        fs::create_dir_all(runs_dir)?;
        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
        let run_id = format!("{date}_run{}_{label}", next_run_number(runs_dir, &date)?);
        let dir = runs_dir.join(&run_id);
        fs::create_dir_all(&dir)?;
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("steps.jsonl"))?;
        Ok(Self {
            run_id,
            dir,
            steps: BufWriter::new(f),
        })
    }

    pub fn step(&mut self, rec: &StepRecord) -> Result<()> {
        serde_json::to_writer(&mut self.steps, rec)?;
        self.steps.write_all(b"\n")?;
        self.steps.flush()?;
        Ok(())
    }

    pub fn write_json<T: Serialize>(&self, name: &str, value: &T) -> Result<PathBuf> {
        let p = self.dir.join(name);
        fs::write(&p, serde_json::to_string_pretty(value)?)?;
        Ok(p)
    }
}

/// Runs are numbered per local day: `2026-10-18_run2_*` is that day's second run.
fn next_run_number(runs_dir: &Path, date: &str) -> Result<u32> {
    let prefix = format!("{date}_run");
    let mut max = 0;
    for entry in fs::read_dir(runs_dir)? {
        let name = entry?.file_name();
        let n = name
            .to_str()
            .and_then(|s| s.strip_prefix(&prefix))
            .and_then(|rest| rest.split('_').next())
            .and_then(|n| n.parse::<u32>().ok());
        if let Some(n) = n {
            max = max.max(n);
        }
    }
    Ok(max + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_runs_per_day() {
        let base = std::env::temp_dir().join(format!("brain-runs-test-{}", now_ms()));
        fs::create_dir_all(base.join("2026-10-18_run1_ask")).unwrap();
        fs::create_dir_all(base.join("2026-10-18_run9_eval-dev-text")).unwrap();
        fs::create_dir_all(base.join("2026-10-17_run12_ask")).unwrap();
        assert_eq!(next_run_number(&base, "2026-10-18").unwrap(), 10);
        assert_eq!(next_run_number(&base, "2026-10-19").unwrap(), 1);
        fs::remove_dir_all(&base).unwrap();
    }
}
