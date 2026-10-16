//! The loop: propose, gate, execute, observe. Repeat until a terminal decision or the step cap.

use crate::decision::{Decision, Proposal, ToolName};
use crate::gate::Gate;
use crate::llm::{extract_json_object, ChatMessage, Llm};
use crate::log::{RunLog, StepRecord};
use crate::tools::Tools;
use anyhow::Result;
use serde::Serialize;
use std::io::{self, BufRead, Write};
use std::time::Instant;

/// What happens when the gate decides Ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AskMode {
    /// Print the question, read a reply from stdin, keep going.
    Interactive,
    /// End the run with outcome Asked (used by the eval).
    Stop,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", content = "text", rename_all = "snake_case")]
pub enum Outcome {
    Answered(String),
    Asked(String),
    Aborted(String),
    StepLimit,
    Error(String),
}

impl Outcome {
    /// The terminal decision this outcome corresponds to, if any.
    pub fn as_decision(&self) -> Option<Decision> {
        match self {
            Outcome::Answered(_) => Some(Decision::Answer),
            Outcome::Asked(_) => Some(Decision::Ask),
            Outcome::Aborted(_) => Some(Decision::Abort),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RunResult {
    pub outcome: Outcome,
    pub steps: usize,
    pub first_decision: Option<Decision>,
    pub decisions: Vec<Decision>,
    pub parse_errors: usize,
    pub overrides: usize,
    pub total_latency_ms: u128,
}

pub struct Agent<'a> {
    pub llm: &'a dyn Llm,
    pub tools: &'a Tools,
    pub gate: &'a dyn Gate,
    pub shadow: Option<&'a dyn Gate>,
    pub max_steps: usize,
    pub ask_mode: AskMode,
    pub verbose: bool,
}

pub fn system_prompt() -> String {
    format!(
        "You are an agent that works one step at a time.\n\
         Reply with exactly one JSON object and nothing else, with these fields:\n\
         thought, decision, tool, input, answer, question.\n\n\
         decision must be one of:\n\
         - \"act\": call one tool. Set tool and input.\n\
         - \"answer\": you have the final answer. Put it in answer.\n\
         - \"ask\": the request refers to something you cannot identify from the request itself (it, them, the file, the total, the number) or is missing a value only the user has. Put your question in question.\n\
         - \"abort\": the request is impossible with these tools, unsafe, or out of scope. Explain in thought.\n\n\
         Tools:\n{}\n\n\
         Rules:\n\
         - Never invent a tool result. Wait for the Observation.\n\
         - Look in the workspace only when the request names a file, folder, or topic. If your thought says you lack context or don't know what \"it\" refers to, the decision is ask, not act. Do not search files hoping to find the missing piece.\n\
         - Use the calculator for any arithmetic, even easy arithmetic.\n\
         - Keep thought to one or two sentences.\n\
         - Unused fields are empty strings. tool is \"none\" unless decision is \"act\".\n\
         - Keep the final answer short: just the value or a one-line reply.",
        Tools::describe()
    )
}

impl Agent<'_> {
    pub fn run(&self, task_id: &str, request: &str, log: &mut RunLog) -> Result<RunResult> {
        let run_id = log.run_id.clone();
        let mut messages = vec![
            ChatMessage::system(system_prompt()),
            ChatMessage::user(request),
        ];
        let mut res = RunResult {
            outcome: Outcome::StepLimit,
            steps: 0,
            first_decision: None,
            decisions: vec![],
            parse_errors: 0,
            overrides: 0,
            total_latency_ms: 0,
        };

        for step in 1..=self.max_steps {
            res.steps = step;
            let started = Instant::now();
            let completion = match self.llm.complete(&messages) {
                Ok(c) => c,
                Err(e) => {
                    res.outcome = Outcome::Error(format!("model call failed: {e:#}"));
                    return Ok(res);
                }
            };
            let latency_ms = started.elapsed().as_millis();
            res.total_latency_ms += latency_ms;
            let raw = completion.content.trim().to_string();
            let reasoning_chars = completion
                .reasoning
                .as_deref()
                .map_or(0, |r| r.chars().count());

            // 1. Parse and validate the proposal.
            let parsed: Result<Proposal, String> = extract_json_object(&raw)
                .ok_or_else(|| "no JSON object found in reply".to_string())
                .and_then(|j| serde_json::from_str::<Proposal>(j).map_err(|e| e.to_string()))
                .and_then(|p| p.validate().map(|_| p));

            let proposal = match parsed {
                Ok(p) => p,
                Err(err) => {
                    res.parse_errors += 1;
                    let obs = format!(
                        "Observation: error: your reply was not a valid step ({err}). Reply with one JSON object."
                    );
                    log.step(&StepRecord {
                        ts_ms: crate::log::now_ms(),
                        run_id: &run_id,
                        task_id,
                        step,
                        latency_ms,
                        reasoning_chars,
                        raw: &raw,
                        proposal: None,
                        parse_error: Some(&err),
                        gate: None,
                        shadow: None,
                        shadow_error: None,
                        observation: Some(&obs),
                        messages: &messages,
                    })?;
                    self.say(step, &format!("parse error: {err}"));
                    messages.push(ChatMessage::assistant(raw));
                    messages.push(ChatMessage::user(obs));
                    continue;
                }
            };

            // 2. Gate it. A shadow gate is logged but never acted on.
            let gate = match self.gate.decide(&messages, &proposal) {
                Ok(g) => g,
                Err(e) => {
                    res.outcome =
                        Outcome::Error(format!("{} gate failed: {e:#}", self.gate.name()));
                    return Ok(res);
                }
            };
            let (shadow, shadow_error) = match self.shadow.map(|s| s.decide(&messages, &proposal)) {
                Some(Ok(g)) => (Some(g), None),
                Some(Err(e)) => (None, Some(format!("{e:#}"))),
                None => (None, None),
            };

            if gate.overridden {
                res.overrides += 1;
            }
            res.first_decision.get_or_insert(gate.decision);
            res.decisions.push(gate.decision);
            self.say(
                step,
                &format!(
                    "{} (proposed {}{}) | {}",
                    gate.decision,
                    proposal.decision,
                    gate.confidence
                        .map(|c| format!(", conf {c:.2}"))
                        .unwrap_or_default(),
                    proposal.thought
                ),
            );

            // 3. Execute the gated decision.
            let mut terminal: Option<Outcome> = None;
            let observation: String = match gate.decision {
                Decision::Act if proposal.tool == ToolName::None => {
                    "Observation: the gate wants a tool step here. Pick a tool.".into()
                }
                Decision::Act => {
                    let out = self.tools.run(proposal.tool, &proposal.input);
                    format!("Observation: {out}")
                }
                Decision::Answer if proposal.answer.trim().is_empty() => {
                    "Observation: the gate wants your final answer now.".into()
                }
                Decision::Answer => {
                    terminal = Some(Outcome::Answered(proposal.answer.trim().to_string()));
                    String::new()
                }
                Decision::Ask => {
                    let q = if proposal.question.trim().is_empty() {
                        "I'm not confident how to proceed. Can you clarify what you want?"
                            .to_string()
                    } else {
                        proposal.question.trim().to_string()
                    };
                    match self.ask_mode {
                        AskMode::Stop => {
                            terminal = Some(Outcome::Asked(q));
                            String::new()
                        }
                        AskMode::Interactive => format!("User reply: {}", read_user_reply(&q)?),
                    }
                }
                Decision::Abort => {
                    terminal = Some(Outcome::Aborted(proposal.thought.trim().to_string()));
                    String::new()
                }
            };

            log.step(&StepRecord {
                ts_ms: crate::log::now_ms(),
                run_id: &run_id,
                task_id,
                step,
                latency_ms,
                reasoning_chars,
                raw: &raw,
                proposal: Some(&proposal),
                parse_error: None,
                gate: Some(&gate),
                shadow: shadow.as_ref(),
                shadow_error: shadow_error.as_deref(),
                observation: (!observation.is_empty()).then_some(observation.as_str()),
                messages: &messages,
            })?;

            if let Some(t) = terminal {
                res.outcome = t;
                return Ok(res);
            }
            if self.verbose && gate.decision == Decision::Act {
                eprintln!("        {}", observation.replace('\n', "\n        "));
            }
            messages.push(ChatMessage::assistant(raw));
            messages.push(ChatMessage::user(observation));
        }

        Ok(res)
    }

    fn say(&self, step: usize, msg: &str) {
        if self.verbose {
            eprintln!("  [{step}] {msg}");
        }
    }
}

fn read_user_reply(question: &str) -> Result<String> {
    print!("\nagent asks: {question}\n> ");
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    Ok(line.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::TextGate;
    use crate::llm::Completion;
    use std::cell::Cell;

    struct Scripted {
        replies: Vec<&'static str>,
        i: Cell<usize>,
    }

    impl Llm for Scripted {
        fn complete(&self, _m: &[ChatMessage]) -> Result<Completion> {
            let i = self.i.get();
            self.i.set(i + 1);
            Ok(Completion {
                content: self.replies.get(i).copied().unwrap_or("{}").into(),
                reasoning: None,
            })
        }
    }

    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn harness(replies: Vec<&'static str>) -> (RunResult, std::path::PathBuf) {
        let base = std::env::temp_dir().join(format!(
            "brain-agent-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(base.join("ws")).unwrap();
        let tools = Tools::new(&base.join("ws")).unwrap();
        let llm = Scripted {
            replies,
            i: Cell::new(0),
        };
        let gate = TextGate;
        let agent = Agent {
            llm: &llm,
            tools: &tools,
            gate: &gate,
            shadow: None,
            max_steps: 5,
            ask_mode: AskMode::Stop,
            verbose: false,
        };
        let mut log = RunLog::create(&base.join("runs"), "test").unwrap();
        (agent.run("t1", "what is 2+3?", &mut log).unwrap(), base)
    }

    #[test]
    fn act_then_answer() {
        let (r, base) = harness(vec![
            r#"{"thought":"add","decision":"act","tool":"calculator","input":"2+3","answer":"","question":""}"#,
            r#"```json
{"thought":"done","decision":"answer","tool":"none","input":"","answer":"5","question":""}
```"#,
        ]);
        assert_eq!(r.outcome, Outcome::Answered("5".into()));
        assert_eq!(r.first_decision, Some(Decision::Act));
        assert_eq!(r.steps, 2);
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn recovers_from_bad_json() {
        let (r, base) = harness(vec![
            "I think the answer is 5",
            r#"{"thought":"oops","decision":"answer","tool":"none","input":"","answer":"5","question":""}"#,
        ]);
        assert_eq!(r.outcome, Outcome::Answered("5".into()));
        assert_eq!(r.parse_errors, 1);
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn ask_stops_in_eval_mode() {
        let (r, base) = harness(vec![
            r#"{"thought":"which file?","decision":"ask","tool":"none","input":"","answer":"","question":"Which file?"}"#,
        ]);
        assert_eq!(r.outcome, Outcome::Asked("Which file?".into()));
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn step_limit() {
        let (r, base) = harness(vec![]);
        assert_eq!(r.outcome, Outcome::StepLimit);
        assert_eq!(r.parse_errors, 5);
        std::fs::remove_dir_all(base).unwrap();
    }
}
