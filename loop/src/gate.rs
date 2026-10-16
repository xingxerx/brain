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
        })
    }
}

#[derive(Serialize)]
struct LensRequest<'a> {
    messages: &'a [ChatMessage],
    proposed: Decision,
}

#[derive(Deserialize)]
struct LensResponse {
    probs: BTreeMap<Decision, f32>,
}

pub struct LensGate {
    http: reqwest::blocking::Client,
    url: String,
    /// Normalized entropy above this means "not sure": the gate returns Ask.
    ask_threshold: f32,
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
        })
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
            })
            .send()
            .with_context(|| {
                format!(
                    "could not reach lens sidecar at {}; is it running?",
                    self.url
                )
            })?;
        if !resp.status().is_success() {
            return Err(anyhow!("lens sidecar returned {}", resp.status()));
        }
        let r: LensResponse = resp
            .json()
            .context("lens sidecar reply did not match the contract")?;
        Ok(apply_entropy_rule(
            r.probs,
            proposal.decision,
            self.ask_threshold,
        ))
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
    fn uncertain_lens_asks() {
        let g = apply_entropy_rule(dist(0.3, 0.3, 0.2, 0.2), Decision::Answer, 0.7);
        assert_eq!(g.decision, Decision::Ask);
        assert!(g.overridden);
    }
}
