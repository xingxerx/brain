//! brain-adapt CLI. Run from the brain repo root. See README.md.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};

use adapt::export::export;
use adapt::ledger::{Ledger, LiveOutcome};
use adapt::retrieve::{format_context, Index};
use adapt::store::{AddOptions, Check, Conflict, CorrectionStore, NewCorrection};

#[derive(Parser)]
#[command(
    name = "brain-adapt",
    about = "Corrections, retrieval and adapter ledger for brain"
)]
struct Cli {
    /// Data folder (corrections.jsonl, adapters.jsonl).
    #[arg(long, global = true)]
    dir: Option<PathBuf>,
    /// PREREG file.
    #[arg(long, global = true, default_value = "adapt/prereg.json")]
    prereg: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Record one correction.
    Add(Box<AddArgs>),
    /// Tombstone a correction (it leaves the next export and adapter).
    Delete {
        id: String,
        #[arg(long)]
        reason: String,
    },
    /// List live corrections.
    List {
        #[arg(long)]
        split: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Counts and the stage 1 go/no-go.
    Stats,
    /// Re-derive every hash in both logs.
    Verify,
    /// Past corrections relevant to a request (train split only).
    Retrieve {
        query: String,
        #[arg(long)]
        k: Option<usize>,
        /// Fraction of the query's content words a hit must contain.
        #[arg(long)]
        min_coverage: Option<f64>,
        /// Print hits and the context block as JSON (used by train/eval_adapt.py).
        #[arg(long)]
        json: bool,
        /// Never return these ids (training uses leave-one-out). Repeatable.
        #[arg(long)]
        exclude: Vec<String>,
    },
    /// Freeze the live set for training and eval.
    Export {
        #[arg(long, default_value = "adapt/data/export")]
        out: PathBuf,
    },
    /// PREREG commands.
    Prereg {
        #[command(subcommand)]
        cmd: PreregCmd,
    },
    /// Adapter registry and promotion gate.
    Ledger {
        #[command(subcommand)]
        cmd: LedgerCmd,
    },
}

#[derive(Subcommand)]
enum PreregCmd {
    /// Record the PREREG's hash. Do this before any stage 2 eval.
    Lock,
    /// Is the PREREG on disk the locked one?
    Status,
}

#[derive(Subcommand)]
enum LedgerCmd {
    Register {
        adapter_dir: PathBuf,
    },
    /// The gate. Baseline = corrections per 100 interactions before this adapter.
    Promote {
        adapter_id: String,
        #[arg(long)]
        verdict: PathBuf,
        #[arg(long)]
        baseline_rate: f64,
        /// You read the review file the eval wrote and the outputs look right.
        #[arg(long)]
        reviewed: bool,
    },
    /// Back to the previous adapter (or the bare model).
    Rollback {
        #[arg(long)]
        reason: String,
    },
    /// Feed live counts since promotion; confirms or rolls back automatically.
    CheckLive {
        #[arg(long)]
        interactions: u64,
        #[arg(long)]
        corrections: u64,
    },
    /// What the loop should serve right now.
    Current {
        #[arg(long)]
        json: bool,
        /// Warn and exit 2 if the current adapter was trained on a different base.
        #[arg(long)]
        base_hash: Option<String>,
    },
}

#[derive(Args)]
struct AddArgs {
    /// Take prompt and system prompt from a `brain dump-prompts` row (exact loop prompt shape).
    #[arg(long, requires = "task_id", conflicts_with_all = ["prompt", "prompt_file", "system_file"])]
    from_dump: Option<PathBuf>,
    #[arg(long)]
    task_id: Option<String>,
    #[arg(long, conflicts_with = "prompt_file")]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    #[arg(long)]
    system_file: Option<PathBuf>,
    /// JSON array of {role, content}: the conversation after the request and before the
    /// corrected step (assistant/user pairs). Omit for a step-1 correction.
    #[arg(long)]
    history_file: Option<PathBuf>,
    /// What the model said (optional but useful for retrieval context).
    #[arg(long, conflicts_with = "model_output_file")]
    model_output: Option<String>,
    #[arg(long)]
    model_output_file: Option<PathBuf>,
    #[arg(long, conflicts_with = "correction_file")]
    correction: Option<String>,
    #[arg(long)]
    correction_file: Option<PathBuf>,
    /// decision | answer | format | other
    #[arg(long)]
    kind: String,
    #[arg(long)]
    check_decision: Option<String>,
    /// Repeatable. Each string must appear in a passing output.
    #[arg(long)]
    check_contains: Vec<String>,
    #[arg(long)]
    check_exact: bool,
    #[arg(long)]
    tag: Option<String>,
    /// Replace these live corrections (they are tombstoned first).
    #[arg(long)]
    supersede: Vec<String>,
    #[arg(long)]
    allow_conflict: bool,
}

fn read_opt(text: Option<String>, file: Option<PathBuf>, what: &str) -> Result<String> {
    match (text, file) {
        (Some(t), _) => Ok(t),
        (None, Some(f)) => {
            std::fs::read_to_string(&f).with_context(|| format!("read {}", f.display()))
        }
        (None, None) => bail!("--{what} or --{what}-file is required"),
    }
}

/// (prompt, system) from a dump-prompts JSONL row: `messages` = [system?, user], or prompt/request + system.
fn from_dump(path: &PathBuf, task: &str) -> Result<(String, Option<String>)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    for line in text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("//"))
    {
        let v: Value = serde_json::from_str(line)?;
        let id = v
            .get("task_id")
            .or_else(|| v.get("id"))
            .and_then(Value::as_str);
        if id != Some(task) {
            continue;
        }
        if let Some(msgs) = v.get("messages").and_then(Value::as_array) {
            let pick = |role: &str| -> Vec<String> {
                msgs.iter()
                    .filter(|m| m.get("role").and_then(Value::as_str) == Some(role))
                    .filter_map(|m| m.get("content").and_then(Value::as_str).map(String::from))
                    .collect()
            };
            let users = pick("user");
            if users.len() != 1 {
                bail!(
                    "task {task}: expected exactly one user message, found {}",
                    users.len()
                );
            }
            let sys = pick("system");
            return Ok((
                users[0].clone(),
                (!sys.is_empty()).then(|| sys.join("\n\n")),
            ));
        }
        let prompt = v
            .get("prompt")
            .or_else(|| v.get("request"))
            .and_then(Value::as_str)
            .with_context(|| format!("task {task}: no messages, prompt or request field"))?;
        return Ok((
            prompt.to_string(),
            v.get("system").and_then(Value::as_str).map(String::from),
        ));
    }
    bail!("task {task} not found in {}", path.display())
}

fn prereg_value(p: &PathBuf) -> Option<Value> {
    std::fs::read_to_string(p)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let dir = adapt::data_dir(cli.dir);
    let store = CorrectionStore::open(&dir);
    let ledger = Ledger::open(&dir);
    let pr = prereg_value(&cli.prereg);
    let num = |ptr: &str, d: f64| {
        pr.as_ref()
            .and_then(|v| v.pointer(ptr))
            .and_then(Value::as_f64)
            .unwrap_or(d)
    };

    match cli.cmd {
        Cmd::Add(a) => {
            let a = *a;
            let checks = [
                a.check_decision.is_some(),
                !a.check_contains.is_empty(),
                a.check_exact,
            ];
            if checks.iter().filter(|x| **x).count() > 1 {
                bail!("pick one check: --check-decision, --check-contains or --check-exact");
            }
            let check = if let Some(d) = a.check_decision {
                Check::Decision { value: d }
            } else if !a.check_contains.is_empty() {
                Check::Contains {
                    value: a.check_contains,
                }
            } else if a.check_exact {
                Check::Exact
            } else {
                eprintln!("note: no check given; this correction will be ungradable in eval");
                Check::None
            };
            let (prompt, system) = match &a.from_dump {
                Some(f) => from_dump(f, a.task_id.as_deref().unwrap_or_default())?,
                None => (
                    read_opt(a.prompt, a.prompt_file, "prompt")?,
                    a.system_file.map(std::fs::read_to_string).transpose()?,
                ),
            };
            if system.is_none() {
                eprintln!("note: no system prompt. The loop always sends one; use --from-dump or --system-file so the adapter trains on the shape it is served");
            }
            let new = NewCorrection {
                prompt,
                system,
                model_output: match (a.model_output, a.model_output_file) {
                    (Some(t), _) => t,
                    (None, Some(f)) => std::fs::read_to_string(&f)
                        .with_context(|| format!("read {}", f.display()))?,
                    (None, None) => String::new(),
                },
                correction: read_opt(a.correction, a.correction_file, "correction")?,
                kind: a.kind,
                check: Some(check),
                tag: a.tag,
                source: "local-cli".into(),
                history: match &a.history_file {
                    Some(f) => serde_json::from_str(
                        &std::fs::read_to_string(f)
                            .with_context(|| format!("read {}", f.display()))?,
                    )
                    .with_context(|| {
                        format!(
                            "{}: expected a JSON array of {{role, content}}",
                            f.display()
                        )
                    })?,
                    None => vec![],
                },
            };
            let opts = AddOptions {
                supersede: a.supersede,
                allow_conflict: a.allow_conflict,
                heldout_frac: num("/heldout_frac", 0.2),
                now: None,
            };
            match store.add(new, &opts) {
                Ok(c) => println!("added {} ({})", c.id, c.split),
                Err(e) => {
                    if let Some(c) = e.downcast_ref::<Conflict>() {
                        for id in &c.with {
                            if let Some(x) = store.get(id)? {
                                eprintln!("  {id}: {:?} -> {:?}", x.prompt, x.correction);
                            }
                        }
                    }
                    return Err(e);
                }
            }
        }
        Cmd::Delete { id, reason } => {
            store.delete(&id, &reason)?;
            println!("deleted {id}");
        }
        Cmd::List { split, json } => {
            for c in store
                .live()?
                .into_iter()
                .filter(|c| split.as_ref().is_none_or(|s| &c.split == s))
            {
                if json {
                    println!("{}", serde_json::to_string(&c)?);
                } else {
                    let p: String = c.prompt.chars().take(70).collect();
                    println!(
                        "{}  {:7}  {:8}  {}",
                        c.id,
                        c.split,
                        c.kind,
                        p.replace('\n', " ")
                    );
                }
            }
        }
        Cmd::Stats => {
            let live = store.live()?;
            let train = live.iter().filter(|c| c.split == "train").count();
            let held = live.iter().filter(|c| c.split == "heldout").count();
            let held_g = live
                .iter()
                .filter(|c| c.split == "heldout" && c.check.is_gradable())
                .count();
            let mut kinds = std::collections::BTreeMap::new();
            for c in &live {
                *kinds.entry(c.kind.clone()).or_insert(0) += 1;
            }
            let min_c = num("/min_corrections", 150.0) as usize;
            let min_h = num("/min_heldout_gradable", 30.0) as usize;
            println!(
                "live corrections: {} (train {}, heldout {}, heldout gradable {})",
                live.len(),
                train,
                held,
                held_g
            );
            let no_sys = live.iter().filter(|c| c.system.is_none()).count();
            if no_sys > 0 {
                println!("warning: {no_sys} corrections have no system prompt (not the shape the loop serves)");
            }
            println!(
                "by kind: {}",
                kinds
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            let go = live.len() >= min_c && held_g >= min_h;
            println!(
                "stage 2 go/no-go: {} (need >= {} corrections and >= {} gradable heldout)",
                if go { "GO" } else { "not yet" },
                min_c,
                min_h
            );
        }
        Cmd::Verify => {
            println!("corrections.jsonl: {} entries OK", store.log().verify()?);
            println!("adapters.jsonl:    {} entries OK", ledger.log().verify()?);
        }
        Cmd::Retrieve {
            query,
            k,
            min_coverage,
            json,
            exclude,
        } => {
            let k = k.unwrap_or(num("/retrieval/k", 3.0) as usize);
            let min_coverage = min_coverage.unwrap_or(num("/retrieval/min_coverage", 0.5));
            let idx = Index::build(&store.live()?);
            let hits = idx.search_excluding(&query, k, min_coverage, &exclude);
            if json {
                println!(
                    "{}",
                    json!({"hits": hits, "context": format_context(&hits), "index_size": idx.len()})
                );
            } else if hits.is_empty() {
                println!(
                    "(no relevant corrections among {} train-split entries)",
                    idx.len()
                );
            } else {
                print!("{}", format_context(&hits));
            }
        }
        Cmd::Export { out } => {
            let m = export(&store, &out)?;
            println!("{}", serde_json::to_string_pretty(&m)?);
        }
        Cmd::Prereg { cmd } => match cmd {
            PreregCmd::Lock => println!("locked {}", ledger.lock_prereg(&cli.prereg)?),
            PreregCmd::Status => {
                let now = adapt::ledger::file_sha(&cli.prereg)?;
                match ledger.locked_prereg()? {
                    Some(l) if l == now => println!("locked and unchanged ({})", &l[..12]),
                    Some(l) => println!(
                        "CHANGED since lock (locked {}, now {})",
                        &l[..12],
                        &now[..12]
                    ),
                    None => println!("not locked (now {})", &now[..12]),
                }
            }
        },
        Cmd::Ledger { cmd } => match cmd {
            LedgerCmd::Register { adapter_dir } => {
                println!("registered {}", ledger.register(&adapter_dir)?)
            }
            LedgerCmd::Promote {
                adapter_id,
                verdict,
                baseline_rate,
                reviewed,
            } => {
                ledger.promote(&adapter_id, &verdict, &cli.prereg, baseline_rate, reviewed)?;
                println!("promoted {adapter_id} (provisional; feed check-live after real use)");
            }
            LedgerCmd::Rollback { reason } => match ledger.rollback(&reason, false)? {
                Some(c) => println!("rolled back; now serving {}", c.adapter_id),
                None => println!("rolled back; now serving the bare model"),
            },
            LedgerCmd::CheckLive {
                interactions,
                corrections,
            } => match ledger.check_live(interactions, corrections, &cli.prereg)? {
                LiveOutcome::KeepObserving { needed } => {
                    println!("keep observing: {needed} more interactions")
                }
                LiveOutcome::Confirmed => println!("confirmed"),
                LiveOutcome::RolledBack { rate, limit } => {
                    println!("ROLLED BACK: {rate:.2} corrections/100 over limit {limit:.2}")
                }
            },
            LedgerCmd::Current { json, base_hash } => {
                let cur = ledger.current()?;
                if json {
                    println!("{}", serde_json::to_string(&cur)?);
                } else {
                    match &cur {
                        Some(c) => println!(
                            "{} ({}) {}",
                            c.adapter_id,
                            if c.confirmed {
                                "confirmed"
                            } else {
                                "provisional"
                            },
                            c.adapter_dir
                        ),
                        None => println!("bare model"),
                    }
                }
                if let (Some(c), Some(b)) = (&cur, base_hash) {
                    if c.base_hash != b {
                        eprintln!("current adapter was trained on base {}, not {}; serve the bare model and retrain", &c.base_hash[..12.min(c.base_hash.len())], b);
                        std::process::exit(2);
                    }
                }
            }
        },
    }
    Ok(())
}
