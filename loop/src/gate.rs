//! Gates decide what the loop actually does with a proposal.
//!
//! - `TextGate` is the baseline: trust the decision the model wrote.
//! - `LensGate` asks the lens sidecar for a distribution over decisions read
//!   from the model's internal state (see sidecar/CONTRACT.md), and applies the
//!   same rule as jspace-lab's intent_gate.py: high entropy means ask.

use crate::decision::{normalized_entropy, Decision, GateOutput, Proposal};
use crate::llm::ChatMessage;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

pub trait Gate {
    fn name(&self) -> &'static str;
    fn decide(&self, messages: &[ChatMessage], proposal: &Proposal) -> Result<GateOutput>;
}

pub struct TextGate;

impl Gate for TextGate {
    fn name(&self) -> &'static str {
        "text"
    }

    fn decide(&self, _messages: &[ChatMessage], proposal: &Proposal) -> Result<GateOutput> {
        Ok(GateOutput {
            decision: proposal.decision,
            confidence: None,
            probs: None,
            entropy: None,
            source: "text".into(),
            overridden: false,
            meta: None,
        })
    }
}

#[derive(Serialize)]
struct LensRequest<'a> {
    messages: &'a [ChatMessage],
    proposed: Decision,
    /// Must match what the generator was sent, or the gate reads a different prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct LensResponse {
    probs: BTreeMap<Decision, f32>,
    /// Everything else the sidecar sends (mode, anchor_mass, margin, timing). Kept for the log.
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

pub struct LensGate {
    http: reqwest::blocking::Client,
    url: String,
    /// Normalized entropy above this means "not sure": the gate returns Ask.
    ask_threshold: f32,
    thinking: Option<bool>,
}

impl LensGate {
    pub fn new(sidecar_url: &str, ask_threshold: f32) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;
        Ok(Self {
            http,
            url: format!("{}/gate", sidecar_url.trim_end_matches('/')),
            ask_threshold,
            thinking: None,
        })
    }

    /// Same setting as the generator's `--no-think`, so the gate renders the same prompt.
    pub fn with_thinking(mut self, thinking: Option<bool>) -> Self {
        self.thinking = thinking;
        self
    }
}

impl Gate for LensGate {
    fn name(&self) -> &'static str {
        "lens"
    }

    fn decide(&self, messages: &[ChatMessage], proposal: &Proposal) -> Result<GateOutput> {
        let resp = self
            .http
            .post(&self.url)
            .json(&LensRequest {
                messages,
                proposed: proposal.decision,
                chat_template_kwargs: self
                    .thinking
                    .map(|t| serde_json::json!({ "enable_thinking": t })),
            })
            .send()
            .with_context(|| {
                format!(
                    "could not reach lens sidecar at {}; is it running?",
                    self.url
                )
            })?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(anyhow!("lens sidecar returned {status}: {body}"));
        }
        let r: LensResponse = resp
            .json()
            .context("lens sidecar reply did not match the contract")?;
        let mut out = apply_entropy_rule(r.probs, proposal.decision, self.ask_threshold);
        if !r.extra.is_empty() {
            out.meta = Some(serde_json::Value::Object(r.extra));
        }
        Ok(out)
    }
}

/// Turn a distribution into a decision. Missing classes count as zero.
pub fn apply_entropy_rule(
    probs: BTreeMap<Decision, f32>,
    proposed: Decision,
    ask_threshold: f32,
) -> GateOutput {
    let ordered: Vec<f32> = Decision::ALL
        .iter()
        .map(|d| *probs.get(d).unwrap_or(&0.0))
        .collect();
    let total: f32 = ordered.iter().sum();
    let entropy = normalized_entropy(&ordered);
    let (top, top_p) = Decision::ALL
        .iter()
        .zip(&ordered)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(d, p)| (*d, if total > 0.0 { p / total } else { 0.0 }))
        .unwrap_or((Decision::Ask, 0.0));

    let decision = if entropy > ask_threshold {
        Decision::Ask
    } else {
        top
    };
    GateOutput {
        decision,
        confidence: Some(top_p),
        probs: Some(probs),
        entropy: Some(entropy),
        source: "lens".into(),
        overridden: decision != proposed,
        meta: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(a: f32, b: f32, c: f32, d: f32) -> BTreeMap<Decision, f32> {
        BTreeMap::from([
            (Decision::Act, a),
            (Decision::Answer, b),
            (Decision::Ask, c),
            (Decision::Abort, d),
        ])
    }

    #[test]
    fn confident_lens_picks_top() {
        let g = apply_entropy_rule(dist(0.9, 0.05, 0.03, 0.02), Decision::Act, 0.7);
        assert_eq!(g.decision, Decision::Act);
        assert!(!g.overridden);
    }

    #[test]
    fn sidecar_extras_are_kept() {
        let r: LensResponse = serde_json::from_str(
            r#"{"probs":{"act":0.9,"answer":0.1},"mode":"probe","margin":2.5,"anchor_mass":0.004}"#,
        )
        .unwrap();
        assert_eq!(r.probs.len(), 2);
        assert_eq!(r.extra["mode"], "probe");
        assert_eq!(r.extra["anchor_mass"], 0.004);
        assert!(!r.extra.contains_key("probs"));
    }

    #[test]
    fn uncertain_lens_asks() {
        let g = apply_entropy_rule(dist(0.3, 0.3, 0.2, 0.2), Decision::Answer, 0.7);
        assert_eq!(g.decision, Decision::Ask);
        assert!(g.overridden);
    }
}
