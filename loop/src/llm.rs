//! Model client. Talks to any OpenAI-compatible server (LM Studio, Ollama, llama.cpp).

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(c: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: c.into(),
        }
    }
    pub fn user(c: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: c.into(),
        }
    }
    pub fn assistant(c: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: c.into(),
        }
    }
}

/// One model reply: the answer text plus any separated reasoning trace.
#[derive(Debug, Clone, Default)]
pub struct Completion {
    pub content: String,
    pub reasoning: Option<String>,
    /// Tokens the server says it generated (thinking included). Used for tokens/s.
    pub completion_tokens: Option<u64>,
    pub prompt_tokens: Option<u64>,
}

pub trait Llm {
    fn complete(&self, messages: &[ChatMessage]) -> Result<Completion>;
}

pub struct OpenAiCompatClient {
    http: reqwest::blocking::Client,
    base_url: String,
    model: String,
    temperature: f32,
    max_tokens: u32,
    /// Ask the server to constrain output to the step JSON schema.
    use_schema: bool,
    /// None: server default. Some(false): ask the chat template to skip the thinking block.
    thinking: Option<bool>,
}

impl OpenAiCompatClient {
    pub fn new(
        base_url: &str,
        model: &str,
        temperature: f32,
        max_tokens: u32,
        use_schema: bool,
    ) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            temperature,
            max_tokens,
            use_schema,
            thinking: None,
        })
    }

    /// `Some(false)` sends `chat_template_kwargs.enable_thinking = false`, which Qwen3.5's
    /// template, the lens sidecar, llama.cpp's server and LM Studio all understand.
    pub fn with_thinking(mut self, thinking: Option<bool>) -> Self {
        self.thinking = thinking;
        self
    }

    fn body(&self, messages: &[ChatMessage], with_schema: bool) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": false,
        });
        if with_schema {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": { "name": "step", "strict": true, "schema": step_schema() }
            });
        }
        if let Some(t) = self.thinking {
            body["chat_template_kwargs"] = json!({ "enable_thinking": t });
        }
        body
    }
}

impl Llm for OpenAiCompatClient {
    fn complete(&self, messages: &[ChatMessage]) -> Result<Completion> {
        let url = format!("{}/chat/completions", self.base_url);
        let resp = self
            .http
            .post(&url)
            .json(&self.body(messages, self.use_schema))
            .send()
            .with_context(|| format!("could not reach {url}; is the model server running?"))?;

        let status = resp.status();
        let text = resp.text()?;
        if !status.is_success() {
            return Err(anyhow!("server returned {status}: {text}"));
        }

        let v: Value = serde_json::from_str(&text).context("server reply was not JSON")?;
        let msg = &v["choices"][0]["message"];
        let content = msg["content"].as_str().unwrap_or_default().to_string();
        let reasoning = msg["reasoning_content"]
            .as_str()
            .or_else(|| msg["reasoning"].as_str())
            .map(str::to_string)
            .filter(|s| !s.is_empty());
        Ok(Completion {
            content,
            reasoning,
            completion_tokens: v["usage"]["completion_tokens"].as_u64(),
            prompt_tokens: v["usage"]["prompt_tokens"].as_u64(),
        })
    }
}

/// JSON schema for one step. Field order matters for constrained decoding:
/// the model writes its thought before committing to a decision.
pub fn step_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "thought":  { "type": "string" },
            "decision": { "type": "string", "enum": ["act", "answer", "ask", "abort"] },
            "tool":     { "type": "string", "enum": ["calculator", "read_file", "list_dir", "none"] },
            "input":    { "type": "string" },
            "answer":   { "type": "string" },
            "question": { "type": "string" }
        },
        "required": ["thought", "decision", "tool", "input", "answer", "question"],
        "additionalProperties": false
    })
}

/// Pull the first top-level JSON object out of free text (for servers or models
/// that ignore the schema and wrap the JSON in prose or code fences).
pub fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, c) in text[start..].char_indices() {
        if in_str {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_from_fenced_text() {
        let t = "sure:\n```json\n{\"a\": \"}{\", \"b\": {\"c\": 1}}\n```\nbye";
        assert_eq!(
            extract_json_object(t),
            Some("{\"a\": \"}{\", \"b\": {\"c\": 1}}")
        );
    }

    #[test]
    fn thinking_flag_goes_in_template_kwargs() {
        let c = OpenAiCompatClient::new("http://x/v1", "m", 0.0, 10, false).unwrap();
        let msgs = [ChatMessage::user("hi")];
        assert!(c.body(&msgs, false).get("chat_template_kwargs").is_none());
        let c = c.with_thinking(Some(false));
        assert_eq!(
            c.body(&msgs, false)["chat_template_kwargs"]["enable_thinking"],
            false
        );
    }

    #[test]
    fn none_when_unbalanced() {
        assert_eq!(extract_json_object("{\"a\": 1"), None);
    }
}
