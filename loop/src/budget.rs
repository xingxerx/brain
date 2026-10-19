//! Search budget: stop the agent from wandering through the workspace.
//!
//! Wraps any gate. When the model proposes another search (`list_dir` or `read_file`) after
//! `budget` searches in a row turned up nothing related to the request, the step is overridden
//! to `answer`, and the loop's normal refill asks the model for the answer text ("it isn't
//! there", or what it did find). Observed failure this fixes: "where is player-harness" opened
//! every folder and file in the workspace and hit the 8-step cap (run 7, 2026-10-18).
//!
//! A search counts as related when its input path or its output contains one of the request's
//! content words (so opening the file the request names is never "fruitless"). Calculator steps
//! are neutral: they neither count nor reset the streak. A user reply resets it.
//!
//! Off unless `--search-budget N` is passed. It changes behavior, so the lens experiment never
//! uses it, and eval runs that do use it are labeled `-budgetN`.

use crate::decision::{Decision, GateOutput, Proposal, ToolName};
use crate::gate::Gate;
use crate::llm::{extract_json_object, ChatMessage};
use anyhow::Result;

/// Function words that never make a search "related" (and short words are ignored anyway).
const STOP: &[&str] = &[
    "the", "and", "for", "with", "what", "where", "which", "who", "how", "does", "are", "was",
    "this", "that", "these", "those", "from", "into", "about", "please", "can", "you", "your",
    "file", "files", "folder", "find", "show", "tell", "give", "there", "have", "has", "all",
];

/// Lowercase content words of length >= 3, split on anything that isn't a letter or digit.
pub fn content_terms(text: &str) -> Vec<String> {
    let mut out: Vec<String> = text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3 && !STOP.contains(w))
        .map(String::from)
        .collect();
    out.sort();
    out.dedup();
    out
}

fn is_search(tool: ToolName) -> bool {
    matches!(tool, ToolName::ListDir | ToolName::ReadFile)
}

/// Consecutive unrelated searches at the end of the conversation.
pub fn fruitless_streak(messages: &[ChatMessage]) -> usize {
    let Some(request) = messages.iter().find(|m| m.role == "user") else {
        return 0;
    };
    let terms = content_terms(&request.content);
    let related = |s: &str| {
        let s = s.to_lowercase();
        terms.iter().any(|t| s.contains(t.as_str()))
    };
    // Pair each tool observation with the proposal that produced it (the latest assistant
    // message before it, which is the refill reply when there was one).
    let mut last: Option<Proposal> = None;
    let mut streak = 0usize;
    let first_user = messages.iter().position(|m| m.role == "user").unwrap_or(0);
    for m in &messages[first_user + 1..] {
        match m.role.as_str() {
            "assistant" => {
                last = extract_json_object(&m.content)
                    .and_then(|j| serde_json::from_str::<Proposal>(j).ok());
            }
            "user" if m.content.starts_with("User reply:") => streak = 0,
            "user" => {
                let Some(obs) = m.content.strip_prefix("Observation: ") else {
                    continue;
                };
                // Parse errors and gate notices are not tool results.
                if obs.starts_with("error: your reply") || obs.starts_with("the gate") {
                    continue;
                }
                let Some(p) = &last else { continue };
                if p.decision != Decision::Act || !is_search(p.tool) {
                    continue;
                }
                if related(&p.input) || related(obs) {
                    streak = 0;
                } else {
                    streak += 1;
                }
            }
            _ => {}
        }
    }
    streak
}

pub struct BudgetGate<'a> {
    pub inner: &'a dyn Gate,
    pub budget: usize,
}

impl Gate for BudgetGate<'_> {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn decide(&self, messages: &[ChatMessage], proposal: &Proposal) -> Result<GateOutput> {
        let mut out = self.inner.decide(messages, proposal)?;
        if out.decision != Decision::Act || !is_search(proposal.tool) {
            return Ok(out);
        }
        let streak = fruitless_streak(messages);
        if streak >= self.budget {
            let mut meta = out.meta.take().unwrap_or_else(|| serde_json::json!({}));
            meta["budget"] = serde_json::json!({
                "fired": true, "streak": streak, "budget": self.budget,
                "inner_decision": out.decision,
            });
            out.meta = Some(meta);
            out.decision = Decision::Answer;
            out.overridden = proposal.decision != Decision::Answer;
            out.source = format!("{}+budget", out.source);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::TextGate;

    fn sys() -> ChatMessage {
        ChatMessage::system("sys")
    }
    fn act(tool: &str, input: &str) -> ChatMessage {
        ChatMessage::assistant(format!(
            r#"{{"thought":"t","decision":"act","tool":"{tool}","input":"{input}","answer":"","question":""}}"#
        ))
    }
    fn obs(s: &str) -> ChatMessage {
        ChatMessage::user(format!("Observation: {s}"))
    }

    #[test]
    fn terms_split_hyphens_and_drop_stopwords() {
        assert_eq!(
            content_terms("Where is player-harness?"),
            vec!["harness", "player"]
        );
    }

    #[test]
    fn player_harness_wander_is_cut_after_two() {
        let mut m = vec![sys(), ChatMessage::user("where is player-harness")];
        assert_eq!(fruitless_streak(&m), 0);
        m.extend([
            act("list_dir", "."),
            obs("config/\ninventory.csv\nnotes.txt"),
        ]);
        assert_eq!(fruitless_streak(&m), 1);
        m.extend([act("list_dir", "config"), obs("settings.toml")]);
        assert_eq!(fruitless_streak(&m), 2);
        let p: Proposal = serde_json::from_str(
            r#"{"thought":"t","decision":"act","tool":"list_dir","input":"recipes","answer":"","question":""}"#,
        )
        .unwrap();
        let g = BudgetGate {
            inner: &TextGate,
            budget: 2,
        };
        let out = g.decide(&m, &p).unwrap();
        assert_eq!(out.decision, Decision::Answer);
        assert!(out.overridden);
        assert_eq!(out.meta.unwrap()["budget"]["streak"], 2);
        let g3 = BudgetGate {
            inner: &TextGate,
            budget: 3,
        };
        assert_eq!(g3.decide(&m, &p).unwrap().decision, Decision::Act);
    }

    #[test]
    fn named_file_and_calculator_do_not_count() {
        let m = vec![
            sys(),
            ChatMessage::user("What is the total value of all stock in inventory.csv?"),
            act("read_file", "inventory.csv"),
            obs("item,qty,unit_price\nwidgets,40,2.50"),
            act("calculator", "40*2.5"),
            obs("100"),
        ];
        assert_eq!(fruitless_streak(&m), 0);
    }

    #[test]
    fn related_output_and_user_reply_reset() {
        let m = vec![
            sys(),
            ChatMessage::user("find the pancake recipe"),
            act("list_dir", "."),
            obs("config/\nrecipes/"),
            act("list_dir", "recipes"),
            obs("omelette.txt\npancakes.txt"),
        ];
        assert_eq!(fruitless_streak(&m), 0);
        let m2 = vec![
            sys(),
            ChatMessage::user("where is it"),
            act("list_dir", "."),
            obs("a/"),
            ChatMessage::user("User reply: the logs"),
            act("list_dir", "logs"),
            obs("app.log"),
        ];
        assert_eq!(fruitless_streak(&m2), 1);
    }

    #[test]
    fn non_search_proposals_pass_through() {
        let m = vec![
            sys(),
            ChatMessage::user("where is player-harness"),
            act("list_dir", "."),
            obs("x"),
            act("list_dir", "y"),
            obs("z"),
        ];
        let p: Proposal = serde_json::from_str(
            r#"{"thought":"t","decision":"answer","tool":"none","input":"","answer":"not here","question":""}"#,
        )
        .unwrap();
        let out = BudgetGate {
            inner: &TextGate,
            budget: 2,
        }
        .decide(&m, &p)
        .unwrap();
        assert_eq!(out.decision, Decision::Answer);
        assert!(!out.overridden);
    }
}
