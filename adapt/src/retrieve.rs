//! Stage 1: corrections retrieved into context. This is also the baseline the
//! adapter has to beat (F2): if pasting the right past corrections into the
//! prompt works as well, the adapter is not worth its risk.
//!
//! BM25 over the prompts of live **train-split** corrections only. Heldout
//! corrections are never retrievable, otherwise eval would be graded on
//! answers it was just shown. Lexical on purpose for v1: no model, no GPU,
//! deterministic, and the Python eval calls this same code through the CLI so
//! both sides retrieve identically.

use std::collections::HashMap;

use serde::Serialize;

use crate::store::{tokens, Correction};

const K1: f64 = 1.2;
/// Function words that would otherwise make unrelated requests match
/// ("in", "the", "what"). Kept short on purpose; content words decide.
const STOP: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "that", "the", "this", "to",
    "was", "what", "with", "you", "your",
];

fn terms(s: &str) -> Vec<String> {
    tokens(s)
        .into_iter()
        .filter(|t| !STOP.contains(&t.as_str()))
        .collect()
}
const B: f64 = 0.75;

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub id: String,
    pub score: f64,
    pub prompt: String,
    pub model_output: String,
    pub correction: String,
    /// For a later-step correction: the observation the model had just seen.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub situation: Option<String>,
}

pub struct Index {
    docs: Vec<(Correction, Vec<String>)>,
    df: HashMap<String, usize>,
    avg_len: f64,
}

impl Index {
    /// Builds from the live set, keeping only split == "train".
    pub fn build(live: &[Correction]) -> Self {
        let docs: Vec<(Correction, Vec<String>)> = live
            .iter()
            .filter(|c| c.split == "train")
            .map(|c| (c.clone(), terms(&c.prompt)))
            .collect();
        let mut df = HashMap::new();
        for (_, t) in &docs {
            let mut seen: Vec<&String> = t.iter().collect();
            seen.sort();
            seen.dedup();
            for w in seen {
                *df.entry(w.clone()).or_insert(0) += 1;
            }
        }
        let avg_len = if docs.is_empty() {
            1.0
        } else {
            docs.iter().map(|(_, t)| t.len()).sum::<usize>() as f64 / docs.len() as f64
        };
        Self {
            docs,
            df,
            avg_len: avg_len.max(1.0),
        }
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    fn idf(&self, w: &str) -> f64 {
        let n = self.docs.len() as f64;
        let d = *self.df.get(w).unwrap_or(&0) as f64;
        ((n - d + 0.5) / (d + 0.5) + 1.0).ln()
    }

    /// Top `k` hits, best BM25 first, ties by id. A hit must contain at least
    /// `min_coverage` of the query's content words. Coverage, not a raw BM25
    /// cutoff, decides relevance: BM25 scores are tiny while the store is small
    /// (every word is in every document), so an absolute cutoff returned
    /// nothing for the first corrections. Found in the loop test.
    pub fn search(&self, query: &str, k: usize, min_coverage: f64) -> Vec<Hit> {
        self.search_excluding(query, k, min_coverage, &[])
    }

    /// Same, never returning the ids in `exclude`. Training uses this to build
    /// each example's context without the example's own correction in it
    /// (leave-one-out), so the adapter trains on the prompt shape it is served
    /// (context + request) without learning to copy its answer from context.
    pub fn search_excluding(
        &self,
        query: &str,
        k: usize,
        min_coverage: f64,
        exclude: &[String],
    ) -> Vec<Hit> {
        let mut q = terms(query);
        q.sort();
        q.dedup();
        let mut hits: Vec<Hit> = self
            .docs
            .iter()
            .filter(|(c, _)| !exclude.contains(&c.id))
            .filter_map(|(c, t)| {
                let len = t.len() as f64;
                let matched = q.iter().filter(|w| t.contains(w)).count();
                let coverage = if q.is_empty() {
                    0.0
                } else {
                    matched as f64 / q.len() as f64
                };
                let mut score = 0.0;
                for w in &q {
                    let tf = t.iter().filter(|x| *x == w).count() as f64;
                    if tf > 0.0 {
                        score += self.idf(w) * tf * (K1 + 1.0)
                            / (tf + K1 * (1.0 - B + B * len / self.avg_len));
                    }
                }
                (coverage >= min_coverage && matched > 0).then(|| Hit {
                    id: c.id.clone(),
                    score: (score * 1e4).round() / 1e4,
                    prompt: c.prompt.clone(),
                    model_output: c.model_output.clone(),
                    correction: c.correction.clone(),
                    situation: c.history.last().map(|m| m.content.clone()),
                })
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap()
                .then_with(|| a.id.cmp(&b.id))
        });
        hits.truncate(k);
        hits
    }
}

fn clip(s: &str, n: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(n).collect::<String>())
    }
}

/// The block injected into the system prompt. Empty when there are no hits,
/// so a no-hit request is byte-identical to the bare model's prompt.
pub fn format_context(hits: &[Hit]) -> String {
    if hits.is_empty() {
        return String::new();
    }
    let mut s = String::from(
        "Past corrections from the user. Apply them when the current request is the same kind of case:\n",
    );
    for h in hits {
        s.push_str(&format!("- Request: {}\n", clip(&h.prompt, 400)));
        if let Some(sit) = &h.situation {
            s.push_str(&format!("  Mid-task, right after: {}\n", clip(sit, 300)));
        }
        s.push_str(&format!(
            "  Wrong reply: {}\n  Correct reply: {}\n",
            clip(&h.model_output, 300),
            clip(&h.correction, 600)
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Check;

    fn c(id: &str, prompt: &str, split: &str) -> Correction {
        Correction {
            id: id.into(),
            ts: 0,
            prompt: prompt.into(),
            system: None,
            model_output: "wrong".into(),
            correction: "right".into(),
            kind: "other".into(),
            check: Check::None,
            tag: None,
            source: "local-cli".into(),
            split: split.into(),
            prompt_hash: String::new(),
            history: vec![],
        }
    }

    #[test]
    fn finds_relevant_and_skips_heldout() {
        let live = vec![
            c("a", "delete every file in the fixtures folder", "train"),
            c("b", "what is 17 times 23", "train"),
            c(
                "h",
                "delete every file in the fixtures folder now",
                "heldout",
            ),
        ];
        let idx = Index::build(&live);
        assert_eq!(idx.len(), 2);
        let hits = idx.search("please delete all files in fixtures", 3, 0.0);
        assert_eq!(hits[0].id, "a");
        assert!(hits.iter().all(|h| h.id != "h"));
        assert!(idx.search("weather in paris", 3, 0.0).is_empty());
        // One-document store: BM25 is tiny, coverage still finds it.
        let one = Index::build(&live[..1]);
        assert_eq!(
            one.search("summarize every file in fixtures folder", 3, 0.5)
                .len(),
            1
        );
        assert!(one
            .search("what is the weather today in fixtures", 3, 0.5)
            .is_empty());
        let ex = idx.search_excluding(
            "please delete all files in fixtures",
            3,
            0.0,
            &["a".to_string()],
        );
        assert!(ex.iter().all(|h| h.id != "a"));
    }

    #[test]
    fn empty_context_when_no_hits() {
        assert_eq!(format_context(&[]), "");
    }
}
