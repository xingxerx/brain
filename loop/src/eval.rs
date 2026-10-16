//! Pre-registered eval: fixed tasks, fixed scoring, results written next to the run log.

use crate::agent::{Agent, Outcome, RunResult};
use crate::decision::Decision;
use crate::log::RunLog;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Dev,
    Heldout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Match {
    /// Expected text appears in the answer (case and punctuation insensitive).
    #[default]
    Contains,
    /// First number in the answer equals the expected number.
    Number,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub split: Split,
    pub prompt: String,
    /// The right first move: act, answer, ask, or abort.
    pub expected_first: Decision,
    /// For tasks that should end in an answer.
    #[serde(default)]
    pub expected_answer: Option<String>,
    #[serde(default, rename = "match")]
    pub match_kind: Match,
}

impl Task {
    /// The terminal decision this task should end in.
    pub fn expected_terminal(&self) -> Decision {
        match self.expected_first {
            Decision::Act => Decision::Answer,
            d => d,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TaskScore {
    pub id: String,
    pub split: Split,
    pub expected_first: Decision,
    pub got_first: Option<Decision>,
    pub first_correct: bool,
    pub success: bool,
    pub result: RunResult,
}

#[derive(Debug, Default, Serialize)]
pub struct Summary {
    pub run_id: String,
    pub gate: String,
    pub model: String,
    pub tasks: usize,
    pub first_decision_accuracy: f64,
    pub success_rate: f64,
    pub per_class_first_accuracy: BTreeMap<String, f64>,
    /// confusion[expected][got] over first decisions ("none" = never reached a decision).
    pub confusion: BTreeMap<String, BTreeMap<String, usize>>,
    pub parse_errors: usize,
    pub gate_overrides: usize,
    pub errors: usize,
    pub step_limits: usize,
    pub mean_steps: f64,
    pub mean_latency_ms: f64,
}

pub fn load_tasks(path: &Path) -> Result<Vec<Task>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut tasks = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let t: Task =
            serde_json::from_str(line).with_context(|| format!("{}:{}", path.display(), n + 1))?;
        tasks.push(t);
    }
    let mut ids: Vec<&str> = tasks.iter().map(|t| t.id.as_str()).collect();
    ids.sort_unstable();
    if let Some(w) = ids.windows(2).find(|w| w[0] == w[1]) {
        bail!("duplicate task id {}", w[0]);
    }
    Ok(tasks)
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn first_number(s: &str) -> Option<f64> {
    let cleaned: String = s.replace(',', "");
    let mut buf = String::new();
    for c in cleaned.chars() {
        if c.is_ascii_digit() || (c == '.' && !buf.is_empty()) || (c == '-' && buf.is_empty()) {
            buf.push(c);
        } else if !buf.is_empty() && buf != "-" {
            break;
        } else {
            buf.clear();
        }
    }
    buf.trim_end_matches('.').parse().ok()
}

pub fn answer_matches(task: &Task, answer: &str) -> bool {
    let Some(exp) = &task.expected_answer else {
        return true;
    };
    match task.match_kind {
        Match::Contains => normalize(answer).contains(&normalize(exp)),
        Match::Number => match (first_number(answer), exp.trim().parse::<f64>()) {
            (Some(a), Ok(e)) => (a - e).abs() <= 1e-6 * e.abs().max(1.0),
            _ => false,
        },
    }
}

pub fn score(task: &Task, result: RunResult) -> TaskScore {
    let first_correct = result.first_decision == Some(task.expected_first);
    let success = match (&result.outcome, task.expected_terminal()) {
        (Outcome::Answered(a), Decision::Answer) => answer_matches(task, a),
        (o, d) => o.as_decision() == Some(d),
    };
    TaskScore {
        id: task.id.clone(),
        split: task.split,
        expected_first: task.expected_first,
        got_first: result.first_decision,
        first_correct,
        success,
        result,
    }
}

pub fn summarize(run_id: &str, gate: &str, model: &str, scores: &[TaskScore]) -> Summary {
    let n = scores.len().max(1) as f64;
    let mut s = Summary {
        run_id: run_id.into(),
        gate: gate.into(),
        model: model.into(),
        tasks: scores.len(),
        ..Default::default()
    };
    let mut per_class: BTreeMap<Decision, (usize, usize)> = BTreeMap::new();
    let (mut first_ok, mut success, mut steps, mut latency) = (0usize, 0usize, 0usize, 0u128);

    for sc in scores {
        first_ok += sc.first_correct as usize;
        success += sc.success as usize;
        steps += sc.result.steps;
        latency += sc.result.total_latency_ms;
        s.parse_errors += sc.result.parse_errors;
        s.gate_overrides += sc.result.overrides;
        match sc.result.outcome {
            Outcome::Error(_) => s.errors += 1,
            Outcome::StepLimit => s.step_limits += 1,
            _ => {}
        }
        let e = per_class.entry(sc.expected_first).or_default();
        e.0 += sc.first_correct as usize;
        e.1 += 1;
        let got = sc.got_first.map_or("none".to_string(), |d| d.to_string());
        *s.confusion
            .entry(sc.expected_first.to_string())
            .or_default()
            .entry(got)
            .or_default() += 1;
    }

    s.first_decision_accuracy = first_ok as f64 / n;
    s.success_rate = success as f64 / n;
    s.mean_steps = steps as f64 / n;
    s.mean_latency_ms = latency as f64 / n;
    s.per_class_first_accuracy = per_class
        .into_iter()
        .map(|(d, (ok, tot))| (d.to_string(), ok as f64 / tot as f64))
        .collect();
    s
}

pub fn run_eval(agent: &Agent, tasks: &[Task], log: &mut RunLog, model: &str) -> Result<Summary> {
    let mut scores = Vec::with_capacity(tasks.len());
    for (i, task) in tasks.iter().enumerate() {
        eprintln!("[{}/{}] {} : {}", i + 1, tasks.len(), task.id, task.prompt);
        let result = agent.run(&task.id, &task.prompt, log)?;
        let sc = score(task, result);
        eprintln!(
            "  -> {} | first {} (expected {}) | {:?}\n",
            if sc.success { "PASS" } else { "FAIL" },
            sc.got_first.map_or("none".into(), |d| d.to_string()),
            sc.expected_first,
            sc.result.outcome
        );
        scores.push(sc);
    }
    log.write_json("scores.json", &scores)?;
    let summary = summarize(&log.run_id, agent.gate.name(), model, &scores);
    log.write_json("summary.json", &summary)?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(exp: &str, m: Match) -> Task {
        Task {
            id: "t".into(),
            split: Split::Dev,
            prompt: String::new(),
            expected_first: Decision::Act,
            expected_answer: Some(exp.into()),
            match_kind: m,
        }
    }

    #[test]
    fn number_matching() {
        let t = task("5000", Match::Number);
        assert!(answer_matches(&t, "The total is 5,000."));
        assert!(answer_matches(&t, "5000"));
        assert!(!answer_matches(&t, "500"));
        let neg = task("-3.5", Match::Number);
        assert!(answer_matches(&neg, "It's -3.5 degrees"));
    }

    #[test]
    fn contains_matching() {
        let t = task("Maple Street", Match::Contains);
        assert!(answer_matches(&t, "It's on maple street!"));
        assert!(!answer_matches(&t, "Oak Avenue"));
    }

    #[test]
    fn bundled_tasks_parse() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../eval/tasks.jsonl");
        let tasks = load_tasks(&p).unwrap();
        assert!(tasks.len() >= 30);
        for d in Decision::ALL {
            assert!(
                tasks.iter().any(|t| t.expected_first == d),
                "no task for {d}"
            );
        }
    }
}
