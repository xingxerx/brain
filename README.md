# brain

v1 of a local agent loop where every step is a **typed decision**: `act`, `answer`, `ask`, or `abort`.
The model proposes a step as schema-constrained JSON, a **gate** decides what actually happens,
tools run in a sandbox, and every step is logged to JSONL with the full state the model saw.

The research question it exists to answer: does a gate that reads the model's internal state
pick the right move more often than the model's own stated decision? See `eval/PREREG.md`.

## Layout

```
loop/      Rust crate (binary: brain)
  src/decision.rs   Decision, Proposal, GateOutput types
  src/llm.rs        OpenAI-compatible client (LM Studio, Ollama, llama.cpp, the sidecar)
  src/gate.rs       TextGate (baseline) and LensGate (sidecar client)
  src/tools.rs      calculator, read_file, list_dir (sandboxed)
  src/agent.rs      the loop, including the one-shot refill when the gate overrides
  src/eval.rs       task loading, scoring, summary, stop-on-errors
  src/log.rs        JSONL run log
  src/report.rs     failure taxonomy and speed for a finished run
eval/
  tasks.jsonl       74 labeled tasks (24 dev, 50 heldout)
  tasks_probe.jsonl 120 probe-training tasks (split "probe": never dev, never heldout)
  fixtures/         the only folder the file tools can see
  PREREG.md         draft pre-registration, lock before heldout
sidecar/
  CONTRACT.md       the /gate API
  engine.py         shared model loading, prompt rendering, layer readout, probe math
  server.py         Qwen3.5-4B generator + gate in one process (probe or anchor mode)
  probe.py          collect activations, train the probe (CV), score it
runs/               logs, results, activations (gitignored)
```

## Quick start (PowerShell, from this folder)

Text gate on LM Studio:

```powershell
lms server start
```

```powershell
cargo run --release -- check
```

```powershell
cargo run --release -- ask "What is the total value of all stock in inventory.csv?"
```

```powershell
cargo run --release -- eval --split dev
```

Results land in `runs\<id>\`: `steps.jsonl` (every step), `scores.json` (per task), `summary.json`.
Run ids are `<date>_run<N>_<kind>`, numbered per local day.

```powershell
cargo run --release -- report runs\<id>
```

`eval` refuses to start if the model server (or, with a lens gate, the sidecar) is not reachable,
and stops after 2 infrastructure errors in a row. A stopped run has `"aborted"` in `summary.json`
and exits with an error: it is not a result.

## The probe gate (v1)

The v0 anchor-word gate scored 6/24 on dev (run 6, 2026-10-18). The late layers do carry signal
(per-class AUC 0.84 to 0.97 except ask), but the hand-picked anchor words scored it badly. v1
replaces the anchors with a linear probe trained on the probe split only.

1. Export the exact step-1 prompts the loop sends:

```powershell
cargo run --release -- dump-prompts --split probe --out runs\prompts_probe.jsonl
```

```powershell
cargo run --release -- dump-prompts --tasks eval\tasks.jsonl --split dev --out runs\prompts_dev.jsonl
```

2. Collect activations (one forward pass per prompt, no generation, about a minute). Use the
   jspace-lab venv and stop the sidecar first, since both need the GPU:

```powershell
$env:QWEN_PATH = "C:\models\Qwen3.5-4B"
python sidecar\probe.py collect --prompts runs\prompts_probe.jsonl --out runs\acts_probe.npz
```

```powershell
python sidecar\probe.py collect --prompts runs\prompts_dev.jsonl --out runs\acts_dev.npz
```

3. Train (5-fold CV over layers 12-31, the 20-30 band, and 4 regularization strengths) and score dev:

```powershell
python sidecar\probe.py train --acts runs\acts_probe.npz --out sidecar\probe.json
```

```powershell
python sidecar\probe.py score --probe sidecar\probe.json --acts runs\acts_dev.npz
```

`train` prints three baselines on the same folds: majority class, **bag-of-words on the request
text**, and shuffled labels. The probe only means something if it beats bag-of-words. For reference,
bag-of-words trained on the probe prompts already gets 0.51 CV on the probe split and **20/24 on dev**,
so dev is mostly separable by its words alone.

4. Serve the probe gate and run the paired dev comparison:

```powershell
python sidecar\server.py --probe sidecar\probe.json
```

```powershell
$S = "--base-url", "http://127.0.0.1:8765/v1", "--model", "Qwen/Qwen3.5-4B", "--no-schema"
cargo run --release -- @S --shadow-lens eval --split dev
```

```powershell
cargo run --release -- @S --gate lens eval --split dev
```

When the lens gate overrides the model and the model's reply has nothing for the gated decision
(an answer with no answer text, an act with no tool), the loop asks the model once to fill it in.
That reply is not gated again. It is logged as `refill_raw`.

## Speed

Measured on the laptop (RTX 5060, 8 GB), sidecar `int8-t0` (already the ~2x faster int8 variant),
thinking on: about **12 tokens/s** and about
**170 tokens per step** (roughly 100 of them thinking), so about 15 s per step. Generation is about
98% of wall time. The gate itself takes about 0.2 s. The levers, biggest first:

**1. Thinking off.** Cuts each step to roughly a third of the tokens. It changes the experimental
condition, so the probe has to be collected with the same setting and runs with and without it
must never be mixed. The sidecar refuses a probe gate request whose thinking setting doesn't
match the probe's.

```powershell
python sidecar\probe.py collect --no-think --prompts runs\prompts_probe.jsonl --out runs\acts_probe_nothink.npz
```

```powershell
cargo run --release -- @S --no-think --shadow-lens eval --split dev
```

**2. Fused kernels for the DeltaNet layers.** 24 of Qwen3.5's 32 blocks are Gated DeltaNet. Without
`flash-linear-attention`, transformers runs them as a slow PyTorch loop. The sidecar prints which
kernels it found at startup. On Windows, flash-linear-attention needs Triton, which comes from the
`triton-windows` package; pick the version that matches your torch (its README has the table):

```powershell
python -c "import torch; print(torch.__version__)"
```

```powershell
pip install triton-windows
```

```powershell
pip install -U "flash-linear-attention>=0.4.2"
```

The speedup has not been measured yet. Compare the `tok/s decode` line the sidecar prints per
request before and after. `causal-conv1d` usually fails to build on Windows; skip it.

**3. Fast interactive mode (no lens).** For using the brain rather than running the experiment,
run Qwen3.5-4B as a GGUF in LM Studio (llama.cpp with CUDA graphs) and use the text gate. Expect
several times the sidecar's speed. Don't run it next to the sidecar on an 8 GB card: both models
won't fit.

```powershell
lms ls
```

```powershell
cargo run --release -- --model <qwen3.5-4b id from lms ls> ask "What is 17 * 23?"
```

**4. bf16 on a 16 GB GPU.** bf16 is the quantization spike's own reference, so it is the cleanest
precision for the experiment, and it skips bitsandbytes' int8 kernels entirely (speed unmeasured).
It needs about 10 GB of VRAM. The probe has to be retrained on bf16 activations (the server checks).

```powershell
python sidecar\server.py --precision bf16 --probe sidecar\probe_bf16.json
```

## Options

| Flag | Default | What it does |
|---|---|---|
| `--model` | `qwen3-8b` | Model id as the server lists it. Was `google/gemma-4-12b-qat` until Oct 18 2026: the dev text-gate runs 1 to 5 used Gemma, so pass `--model google/gemma-4-12b-qat` to reproduce them |
| `--search-budget` | `2` for `ask`, off otherwise | See Search budget below. `0` turns it off |
| `--base-url` | `http://localhost:1234/v1` | Any OpenAI-compatible server (sidecar: `http://127.0.0.1:8765/v1`) |
| `--gate` | `text` | `text` trusts the model's decision, `lens` asks the sidecar |
| `--shadow-lens` | off | Log the lens decision next to the text one without acting on it |
| `--ask-threshold` | `0.7` | Lens: normalized entropy above this becomes `ask` |
| `--no-think` | off | Send `enable_thinking=false` to the generator and the gate |
| `--max-steps` | `8` | Step cap per task |
| `--max-tokens` | `2048` | Per reply, includes reasoning tokens |
| `--no-schema` | off | Skip the JSON schema and parse JSON out of free text |

## If something goes wrong

- **`model server not reachable`** before the eval starts: start LM Studio or the sidecar, then `check`.
- **`EVAL STOPPED`**: the server died mid-run. If you started the sidecar with
  `python sidecar\server.py 2>&1 | Tee-Object -FilePath sidecar.log`, the reason is in `sidecar.log`.
- **`lens sidecar returned 400: probe was trained with thinking on`**: the probe and the run disagree
  on `--no-think`. Retrain the probe or match the flag.
- **Lots of `parse error` lines.** Raise `--max-tokens`, or try `--no-schema`.

## Status

- Done: loop, types, tools, logging, eval harness, lens client, refill on override, stop-on-errors,
  probe pipeline, sidecar v1, 29 unit tests.
- Dev results (Qwen3.5-4B int8, thinking on): text gate 24/24, anchor lens gate 6/24.
- Next: collect activations, train the probe, score dev against the text gate and bag-of-words.
- Open question for PREREG: the text gate is at ceiling on dev, so "lens beats text by 10 points"
  can't pass on tasks like these. See the scan notes before locking.

## adapt (learning from corrections)

`adapt/` keeps your corrections and feeds them back in. `brain correct <run>` records what the first
decision should have been, and `brain ask --adapt` puts relevant past corrections in front of the
system prompt. Stage 2 (a LoRA adapter that has to beat that baseline under a locked PREREG) and
stage 3 (gated promotion with automatic rollback) are in `adapt/README.md`.
`eval` and `dump-prompts` never use adapt, so the lens experiment is unaffected.

## Search budget

`--search-budget N` stops the agent from wandering: after N searches in a row (`list_dir` /
`read_file`) turn up nothing related to the request, the next search is overridden to an answer.
On by default for `ask` (budget 2), off for `eval` and everything else, so the lens experiment is
unchanged. `--search-budget 0` turns it off for `ask`; eval runs that set it are labeled `-budgetN`. Opening the file the request names never counts, and calculator steps are
neutral. Replay any finished run to see where it would have fired:

```powershell
cargo run --release -- budget-replay runs\<run folder> --budget 2
```

On the logs so far, a budget of 2 fires only on the two wandering failures (run 7 "where is
player-harness", and ask-04 in runs 3 and 4), and never on a passing dev task.

## Default model

`qwen3-8b` (in LM Studio) is the default since Oct 18 2026. On "where is player-harness" it was
about 10x faster than Gemma 4 12B (14.5 s vs about 150 s) and followed a retrieved correction where
Gemma ignored it. That's one request, so treat it as a working default, not a benchmark result.

