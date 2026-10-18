//! The typed core: what the model proposes, and what the gate decides.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// The four things the loop can do on any step. Nothing else is representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Decision {
    Act,
    Answer,
    Ask,
    Abort,
}

impl Decision {
    pub const ALL: [Decision; 4] = [
        Decision::Act,
        Decision::Answer,
        Decision::Ask,
        Decision::Abort,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Decision::Act => "act",
            Decision::Answer => "answer",
            Decision::Ask => "ask",
            Decision::Abort => "abort",
        }
    }
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Tools the model is allowed to name. `None` is required when not acting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolName {
    Calculator,
    ReadFile,
    ListDir,
    None,
}

/// One step as proposed by the model. Parsed straight from its JSON output,
/// so a misspelled decision or tool fails to deserialize instead of slipping through.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    #[serde(default)]
    pub thought: String,
    pub decision: Decision,
    #[serde(default = "default_tool")]
    pub tool: ToolName,
    #[serde(default)]
    pub input: String,
    #[serde(default)]
    pub answer: String,
    #[serde(default)]
    pub question: String,
}

fn default_tool() -> ToolName {
    ToolName::None
}

impl Proposal {
    /// Structural checks the schema alone can't express.
    pub fn validate(&self) -> Result<(), String> {
        match self.decision {
            Decision::Act if self.tool == ToolName::None => {
                Err("decision is act but tool is none; name a tool or pick another decision".into())
            }
            Decision::Answer if self.answer.trim().is_empty() => {
                Err("decision is answer but answer is empty".into())
            }
            Decision::Ask if self.question.trim().is_empty() => {
                Err("decision is ask but question is empty".into())
            }
            _ => Ok(()),
        }
    }
}

/// What the gate decided, and how sure it was.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateOutput {
    pub decision: Decision,
    /// Probability of the chosen decision, if the gate produces one.
    pub confidence: Option<f32>,
    /// Full distribution over decisions, if the gate produces one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probs: Option<BTreeMap<Decision, f32>>,
    /// Normalized entropy in [0, 1], same convention as jspace-lab's intent_gate.py.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entropy: Option<f32>,
    /// Which gate produced this ("text" or "lens").
    pub source: String,
    /// True when the gate overrode the model's proposed decision.
    pub overridden: bool,
    /// Anything else the gate reported (lens: mode, anchor_mass, probe margin, timing).
    /// Logged as-is so a run can be audited after the fact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

/// Normalized Shannon entropy of a distribution, in [0, 1].
pub fn normalized_entropy(probs: &[f32]) -> f32 {
    let n = probs.len();
    if n <= 1 {
        return 0.0;
    }
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        return 1.0;
    }
    let h: f32 = probs
        .iter()
        .map(|&p| p / total)
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    (h / (n as f32).ln()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_proposal() {
        let p: Proposal = serde_json::from_str(
            r#"{"thought":"t","decision":"act","tool":"calculator","input":"2+2","answer":"","question":""}"#,
        )
        .unwrap();
        assert_eq!(p.decision, Decision::Act);
        assert_eq!(p.tool, ToolName::Calculator);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn rejects_unknown_decision() {
        let r: Result<Proposal, _> = serde_json::from_str(r#"{"decision":"maybe"}"#);
        assert!(r.is_err());
    }

    #[test]
    fn act_without_tool_is_invalid() {
        let p: Proposal = serde_json::from_str(r#"{"decision":"act","tool":"none"}"#).unwrap();
        assert!(p.validate().is_err());
    }

    #[test]
    fn entropy_bounds() {
        assert!(normalized_entropy(&[1.0, 0.0, 0.0, 0.0]) < 1e-6);
        assert!((normalized_entropy(&[0.25; 4]) - 1.0).abs() < 1e-5);
    }
}
