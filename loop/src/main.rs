//! brain v1: typed decision loop over a local model.
//!
//!   cargo run --release -- check
//!   cargo run --release -- ask "What is 17 * 23?"
//!   cargo run --release -- eval --split dev

mod agent;
mod budget;
mod decision;
mod eval;
mod gate;
mod llm;
mod log;
mod report;
mod tools;

use agent::{Agent, AskMode, Outcome};
use anyhow::{bail, Result};
use clap::{Parser, Subcommand, ValueEnum};
use gate::{Gate, LensGate, TextGate};
use llm::OpenAiCompatClient;
use std::path::PathBuf;
use tools::Tools;

#[derive(Parser)]
#[command(
    name = "brain",
    version,
    about = "v1 agent loop with typed decisions and a pluggable gate"
)]
struct Cli {
    /// OpenAI-compatible base URL (LM Studio default shown).
    #[arg(long, global = true, default_value = "http://localhost:1234/v1")]
    base_url: String,

    /// Model id as the server lists it.
    #[arg(long, global = true, default_value = "qwen3-8b")]
    model: String,

    #[arg(long, global = true, default_value_t = 0.0)]
    temperature: f32,

    /// Includes reasoning tokens for thinking models, so keep it generous.
    #[arg(long, global = true, default_value_t = 2048)]
    max_tokens: u32,

    #[arg(long, global = true, default_value_t = 8)]
    max_steps: usize,

    /// Directory the file tools can see. Nothing outside it is reachable.
    #[arg(long, global = true, default_value = "eval/fixtures")]
    workspace: PathBuf,

    #[arg(long, global = true, default_value = "runs")]
    runs_dir: PathBuf,

    /// Which gate acts on decisions.
    #[arg(long, global = true, value_enum, default_value_t = GateKind::Text)]
    gate: GateKind,

    /// Also run the lens gate in shadow (logged, never acted on).
    #[arg(long, global = true)]
    shadow_lens: bool,

    #[arg(long, global = true, default_value = "http://127.0.0.1:8765")]
    sidecar_url: String,

    /// Lens gate: normalized entropy above this means Ask.
    #[arg(long, global = true, default_value_t = 0.7)]
    ask_threshold: f32,

    /// Don't send a JSON schema; parse JSON out of free text instead.
    #[arg(long, global = true)]
    no_schema: bool,

    /// Ask the chat template to skip the thinking block (about 2-3x fewer tokens per step on
    /// Qwen3.5). This changes the experimental condition: don't mix runs with and without it.
    #[arg(long, global = true)]
    no_think: bool,

    /// After this many searches in a row (list_dir/read_file) find nothing related to the
    /// request, the next search is overridden to an answer. Default: 2 for `ask`, off for
    /// everything else (so eval and the lens experiment are unchanged). 0 turns it off.
    /// Eval runs that set it are labeled -budgetN.
    #[arg(long, global = true)]
    search_budget: Option<usize>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum GateKind {
    Text,
    Lens,
}

#[derive(Subcommand)]
enum Cmd {
    /// Check the model server (and the sidecar, if a lens gate is in use).
    Check,
    /// Run one request interactively. If the agent asks, you answer in the terminal.
    Ask {
        request: String,
        /// Put relevant past corrections (adapt/) in front of the system prompt.
        /// Interactive use only: eval and dump-prompts never use it, so the lens
        /// experiment and the probe always see the plain prompt.
        #[arg(long)]
        adapt: bool,
    },
    /// Replay a run log and show where --search-budget would have stopped a search.
    BudgetReplay {
        /// A run folder.
        dir: PathBuf,
        #[arg(long, default_value_t = 2)]
        budget: usize,
    },
    /// Record a correction for a finished `ask` (or eval) run's first step, into adapt/data.
    /// Takes the exact system prompt, request and model reply from the run log.
    Correct {
        /// The run folder, e.g. runs\2026-10-19_run1_ask
        run: PathBuf,
        /// Which task in the run (eval runs hold many; ask runs hold one).
        #[arg(long)]
        task: Option<String>,
        /// Which step was wrong (default 1). For a later step, the correction keeps the whole
        /// conversation up to that step, so the lesson is "after seeing this, do that".
        #[arg(long, default_value_t = 1)]
        step: u64,
        /// What the model should have replied (one JSON proposal for decision fixes).
        #[arg(long, conflicts_with = "correction_file")]
        correction: Option<String>,
        #[arg(long)]
        correction_file: Option<PathBuf>,
        /// decision | answer | format | other
        #[arg(long, default_value = "decision")]
        kind: String,
        #[arg(long)]
        check_decision: Option<String>,
        #[arg(long)]
        check_contains: Vec<String>,
        #[arg(long)]
        check_exact: bool,
        #[arg(long)]
        tag: Option<String>,
        #[arg(long)]
        supersede: Vec<String>,
        #[arg(long)]
        allow_conflict: bool,
    },
    /// Summarize a finished eval run as a failure taxonomy with fix hints.
    Report {
        /// A run folder, e.g. runs/1760000000000-eval-dev-text
        dir: PathBuf,
    },
    /// Write the step-1 conversation for each task as JSONL, exactly as the loop would send it.
    /// The probe script reads this, so the probe sees the same system prompt as the loop.
    DumpPrompts {
        #[arg(long, default_value = "eval/tasks_probe.jsonl")]
        tasks: PathBuf,
        /// dev, heldout, probe, or all.
        #[arg(long, default_value = "probe")]
        split: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Run the pre-registered task set.
    Eval {
        #[arg(long, default_value = "eval/tasks.jsonl")]
        tasks: PathBuf,
        /// dev, heldout, probe, or all. Heldout is meant to be run once, after PREREG is locked.
        #[arg(long, default_value = "dev")]
        split: String,
        /// Only run the first N tasks (for smoke tests).
        #[arg(long)]
        limit: Option<usize>,
    },
}

/// `ask` stops wandering after this many unrelated searches unless --search-budget says otherwise.
const DEFAULT_ASK_BUDGET: usize = 2;

fn main() -> Result<()> {
    let cli = Cli::parse();

    let lens: Option<LensGate> = if cli.gate == GateKind::Lens || cli.shadow_lens {
        Some(
            LensGate::new(&cli.sidecar_url, cli.ask_threshold)?
                .with_thinking(cli.no_think.then_some(false)),
        )
    } else {
        None
    };
    let text = TextGate;
    let gate: &dyn Gate = match cli.gate {
        GateKind::Text => &text,
        GateKind::Lens => lens.as_ref().expect("lens gate constructed"),
    };
    // Everyday use (`ask`) gets the budget by default; measured runs only when asked for.
    let search_budget: Option<usize> = match &cli.cmd {
        Cmd::Ask { .. } => Some(cli.search_budget.unwrap_or(DEFAULT_ASK_BUDGET)),
        _ => cli.search_budget,
    }
    .filter(|b| *b > 0);
    let budgeted = search_budget.map(|b| budget::BudgetGate {
        inner: gate,
        budget: b,
    });
    let gate: &dyn Gate = match &budgeted {
        Some(b) => b,
        None => gate,
    };
    let shadow: Option<&dyn Gate> = match (cli.shadow_lens, cli.gate) {
        (true, GateKind::Text) => lens.as_ref().map(|l| l as &dyn Gate),
        _ => None,
    };

    match &cli.cmd {
        Cmd::Check => check(&cli),
        Cmd::Report { dir } => report::run(dir),
        Cmd::BudgetReplay { dir, budget } => budget_replay(dir, *budget),
        Cmd::DumpPrompts { tasks, split, out } => {
            let chosen = eval::select_split(eval::load_tasks(tasks)?, split)?;
            if chosen.is_empty() {
                bail!("no tasks with split '{split}' in {}", tasks.display());
            }
            let mut text = String::new();
            for t in &chosen {
                let line = serde_json::json!({
                    "id": t.id,
                    "split": t.split,
                    "expected_first": t.expected_first,
                    "messages": [
                        llm::ChatMessage::system(agent::system_prompt()),
                        llm::ChatMessage::user(&t.prompt),
                    ],
                });
                text.push_str(&serde_json::to_string(&line)?);
                text.push('\n');
            }
            if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(out, text)?;
            eprintln!("wrote {} prompts to {}", chosen.len(), out.display());
            Ok(())
        }
        Cmd::Correct {
            run,
            task,
            step,
            correction,
            correction_file,
            kind,
            check_decision,
            check_contains,
            check_exact,
            tag,
            supersede,
            allow_conflict,
        } => correct(
            run,
            task.as_deref(),
            *step,
            correction.clone(),
            correction_file.clone(),
            kind,
            check_decision.clone(),
            check_contains.clone(),
            *check_exact,
            tag.clone(),
            supersede.clone(),
            *allow_conflict,
        ),
        Cmd::Ask { request, adapt } => {
            let llm = client(&cli)?;
            let tools = Tools::new(&cli.workspace)?;
            let agent = Agent {
                llm: &llm,
                tools: &tools,
                gate,
                shadow,
                max_steps: cli.max_steps,
                ask_mode: AskMode::Interactive,
                verbose: true,
            };
            let mut log = log::RunLog::create(
                &cli.runs_dir,
                &format!(
                    "ask{}{}",
                    if *adapt { "-adapt" } else { "" },
                    search_budget
                        .map(|b| format!("-budget{b}"))
                        .unwrap_or_default()
                ),
            )?;
            let context = if *adapt {
                let cfg = ::adapt::RetrievalConfig::from_prereg(std::path::Path::new(
                    "adapt/prereg.json",
                ));
                let inf = ::adapt::for_request(
                    std::path::Path::new(::adapt::DEFAULT_DIR),
                    request,
                    &cfg,
                )?;
                if let Some(w) = &inf.warning {
                    eprintln!("adapt: {w}");
                }
                if inf.hit_ids.is_empty() {
                    eprintln!("adapt: no relevant past corrections ({})", inf.mode);
                } else {
                    eprintln!(
                        "adapt: using corrections {} ({})",
                        inf.hit_ids.join(", "),
                        inf.mode
                    );
                }
                if let Some(a) = &inf.adapter {
                    eprintln!(
                        "adapt: adapter {} is promoted, but this client only does retrieval; \
                         the sidecar has to load the adapter itself (adapt/INTEGRATION.md step 4)",
                        a.adapter_id
                    );
                }
                log.write_json(
                    "adapt.json",
                    &serde_json::json!({"hit_ids": inf.hit_ids, "context": inf.context, "mode": inf.mode,
                        "warning": inf.warning,
                        "adapter": inf.adapter.as_ref().map(|a| &a.adapter_id)}),
                )?;
                inf.context
            } else {
                String::new()
            };
            let r = agent.run_with_context("ask", request, &context, &mut log)?;
            match &r.outcome {
                Outcome::Answered(a) => println!("\n{a}"),
                Outcome::Aborted(why) => println!("\naborted: {why}"),
                Outcome::Asked(q) => println!("\nasked: {q}"),
                Outcome::StepLimit => println!("\nstopped: hit the {}-step cap", cli.max_steps),
                Outcome::Error(e) => println!("\nerror: {e}"),
            }
            eprintln!(
                "\n{} steps, {:.1}s, log: {}",
                r.steps,
                r.total_latency_ms as f64 / 1000.0,
                log.dir.join("steps.jsonl").display()
            );
            Ok(())
        }
        Cmd::Eval {
            tasks,
            split,
            limit,
        } => {
            let mut chosen = eval::select_split(eval::load_tasks(tasks)?, split)?;
            if let Some(n) = limit {
                chosen.truncate(*n);
            }
            preflight(&cli)?;
            let llm = client(&cli)?;
            let tools = Tools::new(&cli.workspace)?;
            let agent = Agent {
                llm: &llm,
                tools: &tools,
                gate,
                shadow,
                max_steps: cli.max_steps,
                ask_mode: AskMode::Stop,
                verbose: true,
            };
            let mut log = log::RunLog::create(
                &cli.runs_dir,
                &format!(
                    "eval-{split}-{}{}",
                    gate.name(),
                    search_budget
                        .map(|b| format!("-budget{b}"))
                        .unwrap_or_default()
                ),
            )?;
            let s = eval::run_eval(&agent, &chosen, &mut log, &cli.model)?;
            println!("{}", serde_json::to_string_pretty(&s)?);
            eprintln!("\nresults: {}", log.dir.display());
            if let Some(why) = &s.aborted {
                bail!("eval stopped early, not a result: {why}");
            }
            Ok(())
        }
    }
}

/// For each step of a finished run where the model proposed a search, the fruitless streak
/// it was proposed at, and whether a budget of `budget` would have overridden it.
fn budget_replay(dir: &std::path::Path, budget: usize) -> Result<()> {
    let path = dir.join("steps.jsonl");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let (mut searches, mut fired, mut tasks_hit) = (0, 0, std::collections::BTreeSet::new());
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let Some(p) = v.get("proposal").filter(|p| !p.is_null()) else {
            continue;
        };
        let p: decision::Proposal = serde_json::from_value(p.clone())?;
        if p.decision != decision::Decision::Act
            || !matches!(
                p.tool,
                decision::ToolName::ListDir | decision::ToolName::ReadFile
            )
        {
            continue;
        }
        let msgs: Vec<llm::ChatMessage> = serde_json::from_value(v["messages"].clone())?;
        let streak = budget::fruitless_streak(&msgs);
        searches += 1;
        let task = v["task_id"].as_str().unwrap_or("?").to_string();
        if streak >= budget {
            fired += 1;
            println!(
                "FIRE  {task:16} step {:>2}  streak {streak}  {:?} {}",
                v["step"], p.tool, p.input
            );
            tasks_hit.insert(task);
        }
    }
    println!(
        "\n{searches} search steps, budget {budget} would override {fired} of them, in {} task(s): {}",
        tasks_hit.len(),
        tasks_hit.into_iter().collect::<Vec<_>>().join(", ")
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn correct(
    run: &std::path::Path,
    task: Option<&str>,
    step: u64,
    correction: Option<String>,
    correction_file: Option<PathBuf>,
    kind: &str,
    check_decision: Option<String>,
    check_contains: Vec<String>,
    check_exact: bool,
    tag: Option<String>,
    supersede: Vec<String>,
    allow_conflict: bool,
) -> Result<()> {
    use ::adapt::store::{AddOptions, Check, CorrectionStore, NewCorrection};
    let path = run.join("steps.jsonl");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let at_step: Vec<serde_json::Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .filter(|v: &serde_json::Value| v["step"].as_u64() == Some(step))
        .filter(|v| task.is_none_or(|t| v["task_id"].as_str() == Some(t)))
        .collect();
    let rec = match at_step.as_slice() {
        [one] => one,
        [] => bail!(
            "no step-{step} record{} in {}",
            task.map(|t| format!(" for task {t}")).unwrap_or_default(),
            path.display()
        ),
        _ => bail!(
            "{} step-{step} records in this run; pick one with --task",
            at_step.len()
        ),
    };
    // messages = [system, user(request), ..history]: exactly what the model saw at this step.
    let msgs = rec["messages"].as_array().cloned().unwrap_or_default();
    if msgs.len() < 2 || msgs[0]["role"] != "system" || msgs[1]["role"] != "user" {
        bail!("step {step}: expected the conversation to start with system, user");
    }
    let text_of = |m: &serde_json::Value| m["content"].as_str().unwrap_or_default().to_string();
    let mut system = text_of(&msgs[0]);
    let prompt = text_of(&msgs[1]);
    let history: Vec<::adapt::store::Msg> = msgs[2..]
        .iter()
        .map(|m| ::adapt::store::Msg {
            role: m["role"].as_str().unwrap_or_default().to_string(),
            content: text_of(m),
        })
        .collect();
    // Train on the plain system prompt: a run made with --adapt already has the
    // retrieved corrections in front, and training adds its own context.
    if let Ok(a) = std::fs::read_to_string(run.join("adapt.json")) {
        let a: serde_json::Value = serde_json::from_str(&a)?;
        if let Some(ctx) = a["context"].as_str().filter(|c| !c.trim().is_empty()) {
            let prefix = format!("{}\n\n", ctx.trim());
            system = system
                .strip_prefix(&prefix)
                .map(String::from)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "run used adapt context but the system prompt doesn't start with it"
                    )
                })?;
        }
    }
    let correction = match (correction, correction_file) {
        (Some(c), _) => c,
        (None, Some(f)) => std::fs::read_to_string(&f)
            .map_err(|e| anyhow::anyhow!("read correction file {}: {e}", f.display()))?,
        (None, None) => bail!("--correction or --correction-file is required"),
    };
    let n_checks = [
        check_decision.is_some(),
        !check_contains.is_empty(),
        check_exact,
    ]
    .iter()
    .filter(|x| **x)
    .count();
    if n_checks > 1 {
        bail!("pick one check: --check-decision, --check-contains or --check-exact");
    }
    let check = if let Some(d) = check_decision {
        Check::Decision { value: d }
    } else if !check_contains.is_empty() {
        Check::Contains {
            value: check_contains,
        }
    } else if check_exact {
        Check::Exact
    } else if kind == "decision" {
        // Default for decision fixes: the check is the decision in the correction itself.
        match ::adapt::store::extract_decision(&correction) {
            Some(d) => Check::Decision { value: d },
            None => bail!("no decision found in the correction; pass --check-decision"),
        }
    } else {
        Check::None
    };
    let heldout_frac = std::fs::read_to_string("adapt/prereg.json")
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v["heldout_frac"].as_f64())
        .unwrap_or(0.2);
    let store = CorrectionStore::open(::adapt::DEFAULT_DIR);
    let c = store.add(
        NewCorrection {
            prompt,
            system: Some(system),
            model_output: rec["raw"].as_str().unwrap_or_default().to_string(),
            correction,
            kind: kind.to_string(),
            check: Some(check),
            tag,
            source: "brain-user".into(),
            history,
        },
        &AddOptions {
            supersede,
            allow_conflict,
            heldout_frac,
            now: None,
        },
    )?;
    println!("added {} ({})", c.id, c.split);
    Ok(())
}

fn client(cli: &Cli) -> Result<OpenAiCompatClient> {
    Ok(OpenAiCompatClient::new(
        &cli.base_url,
        &cli.model,
        cli.temperature,
        cli.max_tokens,
        !cli.no_schema,
    )?
    .with_thinking(cli.no_think.then_some(false)))
}

/// Refuse to start an eval against a server that isn't there, so a dead server never
/// produces a run full of fake FAILs.
fn preflight(cli: &Cli) -> Result<()> {
    let http = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let url = format!("{}/models", cli.base_url.trim_end_matches('/'));
    if let Err(e) = http.get(&url).send().and_then(|r| r.error_for_status()) {
        bail!("model server not reachable at {url} ({e}). Start it, then run `brain check`.");
    }
    if cli.gate == GateKind::Lens || cli.shadow_lens {
        let health = format!("{}/health", cli.sidecar_url.trim_end_matches('/'));
        if let Err(e) = http.get(&health).send().and_then(|r| r.error_for_status()) {
            bail!("lens sidecar not reachable at {health} ({e}). Start sidecar/server.py first.");
        }
    }
    Ok(())
}

fn check(cli: &Cli) -> Result<()> {
    let http = reqwest::blocking::Client::new();
    let url = format!("{}/models", cli.base_url.trim_end_matches('/'));
    match http.get(&url).send().and_then(|r| r.error_for_status()) {
        Ok(r) => {
            let v: serde_json::Value = r.json()?;
            let ids: Vec<&str> = v["data"]
                .as_array()
                .map(|a| a.iter().filter_map(|m| m["id"].as_str()).collect())
                .unwrap_or_default();
            println!("model server OK at {}", cli.base_url);
            println!("models: {}", ids.join(", "));
            if !ids.contains(&cli.model.as_str()) {
                println!(
                    "warning: '{}' is not in that list; pass --model with one of the ids above",
                    cli.model
                );
            }
        }
        Err(e) => println!("model server NOT reachable at {url}: {e}\n  try: lms server start"),
    }
    match Tools::new(&cli.workspace) {
        Ok(_) => println!("workspace OK: {}", cli.workspace.display()),
        Err(e) => println!("workspace problem: {e}"),
    }
    if cli.gate == GateKind::Lens || cli.shadow_lens {
        let health = format!("{}/health", cli.sidecar_url.trim_end_matches('/'));
        match http.get(&health).send().and_then(|r| r.error_for_status()) {
            Ok(r) => println!(
                "lens sidecar OK at {}: {}",
                cli.sidecar_url,
                r.text().unwrap_or_default()
            ),
            Err(e) => println!("lens sidecar NOT reachable at {health}: {e}"),
        }
    }
    Ok(())
}
