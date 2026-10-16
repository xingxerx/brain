//! brain v1: typed decision loop over a local model.
//!
//!   cargo run --release -- check
//!   cargo run --release -- ask "What is 17 * 23?"
//!   cargo run --release -- eval --split dev

mod agent;
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
    #[arg(long, global = true, default_value = "google/gemma-4-12b-qat")]
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
    Ask { request: String },
    /// Summarize a finished eval run as a failure taxonomy with fix hints.
    Report {
        /// A run folder, e.g. runs/1760000000000-eval-dev-text
        dir: PathBuf,
    },
    /// Run the pre-registered task set.
    Eval {
        #[arg(long, default_value = "eval/tasks.jsonl")]
        tasks: PathBuf,
        /// dev, heldout, or all. Heldout is meant to be run once, after PREREG is locked.
        #[arg(long, default_value = "dev")]
        split: String,
        /// Only run the first N tasks (for smoke tests).
        #[arg(long)]
        limit: Option<usize>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let lens: Option<LensGate> = if cli.gate == GateKind::Lens || cli.shadow_lens {
        Some(LensGate::new(&cli.sidecar_url, cli.ask_threshold)?)
    } else {
        None
    };
    let text = TextGate;
    let gate: &dyn Gate = match cli.gate {
        GateKind::Text => &text,
        GateKind::Lens => lens.as_ref().expect("lens gate constructed"),
    };
    let shadow: Option<&dyn Gate> = match (cli.shadow_lens, cli.gate) {
        (true, GateKind::Text) => lens.as_ref().map(|l| l as &dyn Gate),
        _ => None,
    };

    match &cli.cmd {
        Cmd::Check => check(&cli),
        Cmd::Report { dir } => report::run(dir),
        Cmd::Ask { request } => {
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
            let mut log = log::RunLog::create(&cli.runs_dir, "ask")?;
            let r = agent.run("ask", request, &mut log)?;
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
            let all = eval::load_tasks(tasks)?;
            let mut chosen: Vec<_> = match split.as_str() {
                "dev" => all
                    .into_iter()
                    .filter(|t| t.split == eval::Split::Dev)
                    .collect(),
                "heldout" => all
                    .into_iter()
                    .filter(|t| t.split == eval::Split::Heldout)
                    .collect(),
                "all" => all,
                other => bail!("unknown split '{other}' (use dev, heldout, or all)"),
            };
            if let Some(n) = limit {
                chosen.truncate(*n);
            }
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
            let mut log =
                log::RunLog::create(&cli.runs_dir, &format!("eval-{split}-{}", gate.name()))?;
            let s = eval::run_eval(&agent, &chosen, &mut log, &cli.model)?;
            println!("{}", serde_json::to_string_pretty(&s)?);
            eprintln!("\nresults: {}", log.dir.display());
            Ok(())
        }
    }
}

fn client(cli: &Cli) -> Result<OpenAiCompatClient> {
    OpenAiCompatClient::new(
        &cli.base_url,
        &cli.model,
        cli.temperature,
        cli.max_tokens,
        !cli.no_schema,
    )
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
            Ok(_) => println!("lens sidecar OK at {}", cli.sidecar_url),
            Err(e) => println!("lens sidecar NOT reachable at {health}: {e}"),
        }
    }
    Ok(())
}
