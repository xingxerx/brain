//! `brain report <run dir>`: turn a finished eval run into a failure taxonomy.
//!
//! Reads `scores.json` and `steps.jsonl` as plain JSON, so it also works on runs written
//! by older builds. Each task gets exactly one kind. The point is to say what to fix in
//! the harness (prompt, schema, tool errors, loop guards), not to re-score the model.

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    ModelError,
    EmptyReplies,
    ParseLoop,
    RepeatLoop,
    StepLimit,
    WrongFirst,
    WrongAnswer,
    FrictionTool,
    FrictionParse,
    Ok,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::ModelError => "model_error",
            Kind::EmptyReplies => "empty_replies",
            Kind::ParseLoop => "parse_loop",
            Kind::RepeatLoop => "repeat_loop",
            Kind::StepLimit => "step_limit",
            Kind::WrongFirst => "wrong_first_move",
            Kind::WrongAnswer => "wrong_answer",
            Kind::FrictionTool => "ok_after_tool_error",
            Kind::FrictionParse => "ok_after_parse_error",
            Kind::Ok => "ok",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Kind::ModelError => "The model call itself failed. Check the server, the model name, and context size.",
            Kind::EmptyReplies => "Replies were empty while reasoning used tokens: the model ran out of tokens thinking. Raise --max-tokens or reduce thinking.",
            Kind::ParseLoop => "The model never produced a valid step. Check that the server honors the JSON schema, raise --max-tokens, or try --no-schema.",
            Kind::RepeatLoop => "The same tool call was repeated. Add a loop guard that tells the model it already has that result.",
            Kind::StepLimit => "Ran out of steps without a terminal decision. Look at the step-by-step log: is the task too big, or is the model not converging?",
            Kind::WrongFirst => "The first move was wrong. Compare the confusion matrix: act vs answer errors point at the prompt rules, ask errors at missing examples.",
            Kind::WrongAnswer => "Right first move, wrong final answer. Check the tool output the model saw, the answer matcher, and the short-answer rule.",
            Kind::FrictionTool => "A tool returned an error and the model recovered. Rewrite that tool error message so it says how to fix the call.",
            Kind::FrictionParse => "A reply failed to parse and the model recovered. If this is common, the schema is not being enforced.",
            Kind::Ok => "",
        }
    }
}

#[derive(Debug)]
pub struct TaskReport {
    pub id: String,
    pub kind: Kind,
    pub detail: String,
}

#[derive(Debug, Default)]
pub struct Report {
    pub tasks: Vec<TaskReport>,
    pub parse_reasons: BTreeMap<String, usize>,
    pub tool_errors: BTreeMap<String, usize>,
    pub steps: usize,
    pub mean_step_latency_ms: f64,
}

fn s<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

fn clip(text: &str, n: usize) -> String {
    let flat = text.replace('\n', " ");
    if flat.chars().count() <= n {
        flat
    } else {
        format!("{}...", flat.chars().take(n).collect::<String>())
    }
}

/// Group the first `n` chars of an error message so near-identical errors count together.
fn bucket(text: &str) -> String {
    clip(text.trim(), 60)
}

pub fn analyze(scores: &[Value], steps: &[Value]) -> Report {
    let mut by_task: BTreeMap<&str, Vec<&Value>> = BTreeMap::new();
    for st in steps {
        by_task.entry(s(st, "task_id")).or_default().push(st);
    }

    let mut report = Report {
        steps: steps.len(),
        ..Default::default()
    };
    let total_latency: f64 = steps
        .iter()
        .filter_map(|v| v.get("latency_ms").and_then(Value::as_f64))
        .sum();
    if !steps.is_empty() {
        report.mean_step_latency_ms = total_latency / steps.len() as f64;
    }

    for sc in scores {
        let id = s(sc, "id").to_string();
        let mine: &[&Value] = by_task.get(id.as_str()).map(Vec::as_slice).unwrap_or(&[]);
        let success = sc.get("success").and_then(Value::as_bool).unwrap_or(false);
        let first_correct = sc
            .get("first_correct")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let outcome = &sc["result"]["outcome"];
        let okind = s(outcome, "kind");
        let otext = s(outcome, "text");
        let expected = s(sc, "expected_first");
        let got = sc
            .get("got_first")
            .and_then(Value::as_str)
            .unwrap_or("none");

        let mut tool_error_count = 0usize;
        let mut parse_error_count = 0usize;
        let mut empty_with_reasoning = 0usize;
        let mut calls: BTreeMap<(String, String), usize> = BTreeMap::new();
        for st in mine {
            if let Some(err) = st.get("parse_error").and_then(Value::as_str) {
                parse_error_count += 1;
                *report.parse_reasons.entry(bucket(err)).or_default() += 1;
                let reasoning = st
                    .get("reasoning_chars")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                if s(st, "raw").trim().is_empty() && reasoning > 0 {
                    empty_with_reasoning += 1;
                }
            }
            // A parse-error step also carries an "Observation: error:" telling the model its
            // reply was invalid. That is not a tool failure, so skip it here.
            let parse_failed = st.get("parse_error").is_some_and(|v| !v.is_null());
            if let Some(obs) = st.get("observation").and_then(Value::as_str) {
                if let (false, Some(rest)) = (parse_failed, obs.strip_prefix("Observation: error:"))
                {
                    tool_error_count += 1;
                    *report.tool_errors.entry(bucket(rest)).or_default() += 1;
                }
            }
            if let Some(p) = st.get("proposal").filter(|p| !p.is_null()) {
                if s(p, "decision") == "act" {
                    *calls
                        .entry((s(p, "tool").to_string(), s(p, "input").to_string()))
                        .or_default() += 1;
                }
            }
        }
        let repeated = calls.iter().find(|(_, n)| **n >= 3);
        let all_parse_errors = !mine.is_empty() && parse_error_count == mine.len();

        let (kind, detail) = if okind == "error" {
            (Kind::ModelError, clip(otext, 120))
        } else if okind == "step_limit" && empty_with_reasoning > 0 && all_parse_errors {
            (
                Kind::EmptyReplies,
                format!("{empty_with_reasoning} empty replies with reasoning tokens spent"),
            )
        } else if okind == "step_limit" && all_parse_errors {
            (
                Kind::ParseLoop,
                format!("{parse_error_count} steps, none valid"),
            )
        } else if okind == "step_limit" && repeated.is_some() {
            let ((tool, input), n) = repeated.unwrap();
            (
                Kind::RepeatLoop,
                format!("{tool}({}) called {n} times", clip(input, 40)),
            )
        } else if okind == "step_limit" {
            (Kind::StepLimit, format!("{} steps used", mine.len()))
        } else if !first_correct {
            let thought = mine
                .iter()
                .find_map(|st| st.get("proposal").filter(|p| !p.is_null()))
                .map(|p| clip(s(p, "thought"), 90))
                .unwrap_or_default();
            (
                Kind::WrongFirst,
                format!("expected {expected}, got {got}. thought: {thought}"),
            )
        } else if !success {
            (Kind::WrongAnswer, format!("{okind}: {}", clip(otext, 90)))
        } else if tool_error_count > 0 {
            (
                Kind::FrictionTool,
                format!("{tool_error_count} tool error(s)"),
            )
        } else if parse_error_count > 0 {
            (
                Kind::FrictionParse,
                format!("{parse_error_count} parse error(s)"),
            )
        } else {
            (Kind::Ok, String::new())
        };
        report.tasks.push(TaskReport { id, kind, detail });
    }
    report
}

pub fn render(r: &Report, run: &str) -> String {
    let mut out = String::new();
    let mut counts: BTreeMap<Kind, usize> = BTreeMap::new();
    for t in &r.tasks {
        *counts.entry(t.kind).or_default() += 1;
    }
    let _ = writeln!(out, "run: {run}");
    let _ = writeln!(
        out,
        "{} tasks, {} steps, mean step latency {:.0} ms\n",
        r.tasks.len(),
        r.steps,
        r.mean_step_latency_ms
    );
    let _ = writeln!(out, "kinds (worst first):");
    for (k, n) in &counts {
        let _ = writeln!(out, "  {:<22}{n}", k.label());
    }

    for k in counts.keys().filter(|k| **k != Kind::Ok) {
        let _ = writeln!(out, "\n== {} ==", k.label());
        let _ = writeln!(out, "fix: {}", k.hint());
        for t in r.tasks.iter().filter(|t| t.kind == *k) {
            let _ = writeln!(out, "  {:<14}{}", t.id, t.detail);
        }
    }
    if !r.parse_reasons.is_empty() {
        let _ = writeln!(out, "\nparse errors by reason:");
        for (m, n) in &r.parse_reasons {
            let _ = writeln!(out, "  {n:>3}  {m}");
        }
    }
    if !r.tool_errors.is_empty() {
        let _ = writeln!(out, "\ntool errors by message:");
        for (m, n) in &r.tool_errors {
            let _ = writeln!(out, "  {n:>3}  {m}");
        }
    }
    out
}

pub fn run(dir: &Path) -> Result<()> {
    let scores: Vec<Value> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("scores.json")).with_context(|| {
            format!(
                "reading {}/scores.json (was this an eval run?)",
                dir.display()
            )
        })?,
    )?;
    let steps_text = std::fs::read_to_string(dir.join("steps.jsonl"))
        .with_context(|| format!("reading {}/steps.jsonl", dir.display()))?;
    let steps: Vec<Value> = steps_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    print!("{}", render(&analyze(&scores, &steps), &name));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn score(id: &str, success: bool, first_ok: bool, kind: &str) -> Value {
        json!({"id": id, "success": success, "first_correct": first_ok,
               "expected_first": "act", "got_first": "answer",
               "result": {"outcome": {"kind": kind, "text": "x"}}})
    }

    fn step(task: &str, n: usize, extra: Value) -> Value {
        let mut v = json!({"task_id": task, "step": n, "latency_ms": 100});
        v.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        v
    }

    fn kind_of(r: &Report, id: &str) -> Kind {
        r.tasks.iter().find(|t| t.id == id).unwrap().kind
    }

    #[test]
    fn classifies_each_kind() {
        let scores = vec![
            score("ok", true, true, "answered"),
            score("wrong-first", false, false, "answered"),
            score("wrong-ans", false, true, "answered"),
            score("friction", true, true, "answered"),
            score("loop", false, true, "step_limit"),
            score("empty", false, true, "step_limit"),
        ];
        let call = json!({"decision":"act","tool":"read_file","input":"a.txt","thought":"t"});
        let mut steps = vec![
            step(
                "ok",
                1,
                json!({"proposal": {"decision":"answer","tool":"none","input":"","thought":"t"}}),
            ),
            step(
                "wrong-first",
                1,
                json!({"proposal": {"decision":"answer","tool":"none","input":"","thought":"just know it"}}),
            ),
            step("wrong-ans", 1, json!({"proposal": call.clone()})),
            step(
                "friction",
                1,
                json!({"proposal": call.clone(), "observation": "Observation: error: no such file"}),
            ),
        ];
        for n in 1..=3 {
            steps.push(step(
                "loop",
                n,
                json!({"proposal": call.clone(), "observation": "Observation: x"}),
            ));
        }
        for n in 1..=2 {
            steps.push(step("empty", n, json!({"raw": "", "reasoning_chars": 900, "parse_error": "no JSON object found in reply"})));
        }
        let r = analyze(&scores, &steps);
        assert_eq!(kind_of(&r, "ok"), Kind::Ok);
        assert_eq!(kind_of(&r, "wrong-first"), Kind::WrongFirst);
        assert_eq!(kind_of(&r, "wrong-ans"), Kind::WrongAnswer);
        assert_eq!(kind_of(&r, "friction"), Kind::FrictionTool);
        assert_eq!(kind_of(&r, "loop"), Kind::RepeatLoop);
        assert_eq!(kind_of(&r, "empty"), Kind::EmptyReplies);
        assert_eq!(r.tool_errors.len(), 1);
        assert_eq!(r.parse_reasons.values().sum::<usize>(), 2);
    }

    #[test]
    fn parse_error_observation_is_not_a_tool_error() {
        let scores = vec![score("t1", true, true, "answered")];
        let steps = vec![
            step(
                "t1",
                1,
                json!({"parse_error": "decision is act but tool is none", "raw": "{}",
                       "observation": "Observation: error: your reply was not a valid step (x)"}),
            ),
            step(
                "t1",
                2,
                json!({"proposal": {"decision":"answer","tool":"none","input":"","thought":"t"}}),
            ),
        ];
        let r = analyze(&scores, &steps);
        assert_eq!(kind_of(&r, "t1"), Kind::FrictionParse);
        assert!(r.tool_errors.is_empty());
        assert_eq!(r.parse_reasons.len(), 1);
    }

    #[test]
    fn render_lists_failures_with_fix() {
        let scores = vec![score("t1", false, false, "answered")];
        let steps = vec![step(
            "t1",
            1,
            json!({"proposal": {"decision":"answer","tool":"none","input":"","thought":"guess"}}),
        )];
        let text = render(&analyze(&scores, &steps), "run-x");
        assert!(text.contains("wrong_first_move"));
        assert!(text.contains("fix:"));
        assert!(text.contains("t1"));
    }
}
