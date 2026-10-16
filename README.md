# brain

v1 of a local agent loop where every step is a **typed decision**: `act`, `answer`, `ask`, or `abort`.
The model proposes a step as schema-constrained JSON, a **gate** decides what actually happens,
tools run in a sandbox, and every step is logged to JSONL with the full state the model saw.

The research question it exists to answer: does a gate that reads the model's internal state
(the J-lens, from `jspace-lab`) pick the right move more often than the model's own stated decision?
See `eval/PREREG.md`.

## Layout

```
loop/      Rust crate (binary: brain)
  src/decision.rs   Decision, Proposal, GateOutput types
  src/llm.rs        OpenAI-compatible client (LM Studio, Ollama, llama.cpp)
  src/gate.rs       TextGate (baseline) and LensGate (sidecar client)
  src/tools.rs      calculator, read_file, list_dir (sandboxed)
  src/agent.rs      the loop
  src/eval.rs       task loading, scoring, summary
  src/log.rs        JSONL run log
  src/report.rs     failure taxonomy for a finished run
eval/
  tasks.jsonl       74 labeled tasks (24 dev, 50 heldout)
  fixtures/         the only folder the file tools can see
  PREREG.md         draft pre-registration, lock before heldout
sidecar/
  CONTRACT.md       /gate API the lens sidecar must serve (sidecar not built yet)
runs/               logs and results (gitignored)
```

## Quick start (PowerShell, from this folder)

Start the model server (LM Studio):

```powershell
lms server start
```

Check everything is reachable:

```powershell
cargo run --release -- check
```

One request, interactive (if the agent asks a question, you answer in the terminal):

```powershell
cargo run --release -- ask "What is the total value of all stock in inventory.csv?"
```

Smoke test the eval on 3 tasks:

```powershell
cargo run --release -- eval --limit 3
```

Full dev split:

```powershell
cargo run --release -- eval --split dev
```

Results land in `runs\<id>\`: `steps.jsonl` (every step), `scores.json` (per task), `summary.json`.

See what to fix in the harness (failure kinds, fix hints, parse and tool error counts):

```powershell
cargo run --release -- report runs\<id>
```

## Options

| Flag | Default | What it does |
|---|---|---|
| `--model` | `google/gemma-4-12b-qat` | Model id as the server lists it |
| `--base-url` | `http://localhost:1234/v1` | Any OpenAI-compatible server (Ollama: `http://localhost:11434/v1`) |
| `--gate` | `text` | `text` trusts the model's decision, `lens` asks the sidecar |
| `--shadow-lens` | off | Log the lens decision next to the text one without acting on it |
| `--ask-threshold` | `0.7` | Lens: normalized entropy above this becomes `ask` |
| `--max-steps` | `8` | Step cap per task |
| `--max-tokens` | `2048` | Per reply, includes reasoning tokens |
| `--no-schema` | off | Skip the JSON schema and parse JSON out of free text |

## If something goes wrong

- **`could not reach ... is the model server running?`** Run `lms server start`, then `check`.
- **Lots of `parse error` lines.** The server may be ignoring the schema, or the model is running out
  of tokens while thinking. Raise `--max-tokens`, or try `--no-schema`.
- **Very slow.** In LM Studio, reload the model with Parallel 1 and Context 4096.

## Status

- Done: loop, types, tools, logging, eval harness, lens client, 19 unit tests.
- Waiting on the spike: the lens sidecar itself (`sidecar/CONTRACT.md`).
- Current model (Gemma 4 12B) is a stand-in for building and testing. The real experiment runs on Qwen3.5-4B.
