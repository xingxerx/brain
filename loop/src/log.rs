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
    pub reasoning_chars: usize,
    /// The model's raw reply text.
    pub raw: &'a str,
    pub proposal: Option<&'a Proposal>,
    pub parse_error: Option<&'a str>,
    pub gate: Option<&'a GateOutput>,
    /// Shadow gate: logged for comparison, never acted on.
    pub shadow: Option<&'a GateOutput>,
    pub shadow_error: Option<&'a str>,
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
        let run_id = format!("{}-{}", now_ms(), label);
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
