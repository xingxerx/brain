# adapt: Tier 2 learning for brain

brain keeps your corrections. Stage 1 retrieves past corrections into context. Stage 2 tests whether
a LoRA adapter trained on them beats that. Stage 3 serves the adapter only if a locked
pre-registration says it earned it, and rolls it back automatically if real use gets worse.

## What it unlocks

- **Stage 1, available now:** you correct a decision once and the next similar request sees that
  correction. CPU only, no training, no GPU.
- **Stage 2:** an honest answer to whether training on your corrections beats putting them in the
  prompt. That answer can be "no", and then you stop at stage 1.
- **Stage 3:** a brain that improves from use, with every change traceable, reversible and gated.

**What it does not unlock:** new facts (fine-tuning a small model on facts is unreliable, so use
retrieval for those), better reasoning, or thinking-mode training (see F9 below).

## Layout

```
adapt/
  Cargo.toml           Rust crate brain-adapt (lib "adapt" + CLI "brain-adapt")
  src/chain.rs         append-only, hash-chained JSONL (same design as LIMEN RouterMemory)
  src/store.rs         corrections: fixed splits, checks, conflicts, tombstones
  src/retrieve.rs      BM25 over train-split corrections (stage 1 and the baseline)
  src/export.rs        freeze the live set for training and eval
  src/ledger.rs        adapter registry, the promotion gate, live rollback
  prereg.json          every threshold; lock it before any stage 2 eval
  PREREG_ADAPT.md      what the thresholds mean and why
  INTEGRATION.md       how to wire this into loop/ and the sidecar
  train/common.py      rules shared with Rust (parity-tested)
  train/train_lora.py  bf16 LoRA from scratch on the train split
  train/eval_adapt.py  bare / retrieval / adapter / adapter+retrieval, canary, drift, verdict
  train/lens.py        the sidecar probe applied offline (thinking-off, base or adapter)
  train/lens_filter.py lens use 1: pick which train corrections the LoRA trains on
  tests/               shared parity cases, CPU end-to-end smoke
  data/                corrections.jsonl and adapters.jsonl (commit these), export/ (ignored)
```

## Build and test (PowerShell, from the brain repo root)

`adapt` is a member of brain's Cargo workspace, so it builds into the repo's own `target\`. Plain `cargo run` / `cargo build` / `cargo test` act on the loop only (so `cargo run --release -- ask ...` works). Use `--workspace` to include adapt, and `cargo adapt <command>` (an alias in `.cargo/config.toml`) to run the brain-adapt CLI, for example `cargo adapt stats`.

```powershell
cargo build --release --workspace
```

```powershell
cargo test --workspace
```

```powershell
python adapt\train\test_common.py
```

The end-to-end smoke test runs on CPU with a tiny random model (about 2 minutes, nothing downloaded). Use the jspace-lab venv:

```powershell
python adapt\tests\smoke_e2e.py
```

The learning check trains the tiny model hard on the synthetic pattern (about 1 minute). It shows
the gate passes when an adapter really helps, and only the smoke flag blocks it:

```powershell
$env:SMOKE_LEARN = "1"; python adapt\tests\smoke_e2e.py; Remove-Item Env:SMOKE_LEARN
```

Add the binary to the session path so the commands below are short:

```powershell
$env:PATH = "$PWD\target\release;$env:PATH"
```

## Stage 1: capture and retrieve

The easiest way is straight from the loop. Run a request, and if the first decision was wrong, record what it should have been. `brain correct` takes the exact system prompt, request and model reply from the run log:

```powershell
cargo run --release -- ask "Summarize the file"
```

```powershell
cargo run --release -- correct runs\<that run folder> --correction-file fix.json
```

If the mistake came later in the run (the first step was fine, then it went wrong), name the step. The correction keeps the whole conversation up to that step, so the lesson is "after seeing this, do that":

```powershell
cargo run --release -- correct runs\<that run folder> --step 2 --correction-file fix.json
```

For decision fixes, the check is taken from the decision inside your correction. For other kinds, add `--kind answer --check-contains "..."`. Then use past corrections on later requests:

```powershell
cargo run --release -- ask --adapt "Summarize that file for me"
```

`--adapt` only exists on `ask`. `eval` and `dump-prompts` never add corrections, so the lens experiment and the probe always see the plain prompt. A run made with `--adapt` logs the corrections it used in `adapt.json`.

You can also capture straight from a `dump-prompts` file. Export the dev prompts once (brain already has this command):

```powershell
cargo run --release -- dump-prompts --tasks eval\tasks.jsonl --split dev --out runs\prompts_dev.jsonl
```

Then record a correction for a task where the model chose wrong:

```powershell
brain-adapt add --from-dump runs\prompts_dev.jsonl --task-id dev_ask_03 --model-output '{"decision":"act","tool":"read_file"}' --correction '{"decision":"ask","question":"Which file do you mean?"}' --kind decision --check-decision ask
```

JSON in arguments needs PowerShell 7.3 or newer. Windows PowerShell 5.1 strips the inner quotes.
On 5.1, put the JSON in files and use `--model-output-file` and `--correction-file` instead.

For a free-form request, pass the loop's system prompt yourself (`--system-file`), or `add` warns you:

```powershell
brain-adapt add --prompt "Summarize inventory.csv" --system-file runs\system_prompt.txt --correction "Total stock value is 4,210 dollars across 12 items." --kind answer --check-contains "4,210"
```

Each correction needs one check so eval never needs a model to grade it:

| Kind | Check |
|---|---|
| `decision` | `--check-decision act\|answer\|ask\|abort` (required) |
| `answer`, `format`, `other` | `--check-contains <s>` (repeatable), `--check-exact`, or none (then it's ungradable) |

### Retrieval by meaning

Word match only finds near-exact repeats: "where is the LIMEN repo" never reached the
"where is player-harness" correction (run 13). So retrieval also compares **meaning**, using the
embedding model LM Studio already has (`text-embedding-nomic-embed-text-v1.5`). A past correction
is used if it shares half the request's content words **or** its meaning is at least
`retrieval.semantic.min_similarity` close (0.7 to start). Settings are in `prereg.json`:

```json
"semantic": { "model": "text-embedding-nomic-embed-text-v1.5", "url": "http://localhost:1234/v1", "min_similarity": 0.7 }
```

- LM Studio's server must be running (`lms server start`). It loads the embedding model on first use, or load it yourself: `lms load text-embedding-nomic-embed-text-v1.5`.
- If it isn't reachable, `brain ask --adapt` says so and falls back to word match. Training and eval never fall back: they stop, so a run can't silently retrieve differently from what the PREREG says.
- Each correction is embedded once and cached in `data/embeddings.jsonl` (safe to delete).
- Set `"semantic": null` to go back to word match only. Use `BRAIN_EMBED_URL` to point at another server.

**Calibrate the threshold** on your own corrections. 0.7 is a starting guess, not measured. See every stored correction's word overlap and meaning similarity for a request:

```powershell
cargo adapt retrieve --explain "where is the LIMEN repo"
```

```powershell
cargo adapt retrieve --explain "what is 17 times 23"
```

Pick `min_similarity` between the scores of requests that should match and ones that shouldn't. Do it before locking the PREREG.

Other commands:

```powershell
brain-adapt stats
```

```powershell
brain-adapt retrieve "delete everything in fixtures"
```

```powershell
brain-adapt delete c_1234abcd5678 --reason "I was wrong"
```

```powershell
brain-adapt verify
```

- If a new correction nearly matches an existing one but says something different, it's refused.
  Use `--supersede <id>` to replace the old one, or `--allow-conflict` if both are right.
- `stats` prints the stage 2 go/no-go: at least 150 corrections and at least 30 gradable heldout.
  **If you don't reach that in the time box, stop. Stage 1 is the win.**

## Stage 2: one manual experiment (RTX 5080)

1. Fill in the two `TODO`s in `prereg.json`: `serving_precision` (whatever the loop actually serves,
   for example `int8-t0` on the laptop or `bf16` on the 5080) and `lens.probe_sha256` (step 1b).
   Read `PREREG_ADAPT.md`. Commit, then lock:

```powershell
brain-adapt prereg lock
```

```powershell
git add adapt; git commit -m "adapt: lock PREREG"
```

1b. Before locking: the lens. adapt runs thinking off, and the current `sidecar\probe.json` was
trained thinking on (a different readout token), so train a thinking-off probe once. Same probe
split, same CV, about the same time as the original:

```powershell
python sidecar\probe.py collect --no-think --precision int8-t0 --prompts runs\prompts_probe.jsonl --out runs\acts_probe_nothink.npz
```

```powershell
python sidecar\probe.py train --acts runs\acts_probe_nothink.npz --out sidecar\probe_nothink.json
```

Check it still beats bag-of-words in the printout (if not, set `lens.filter` and `lens.eval_report`
to false and record why). Then pin it and put the hash in `prereg.json` as `lens.probe_sha256`:

```powershell
(Get-FileHash sidecar\probe_nothink.json -Algorithm SHA256).Hash.ToLower()
```

2. Freeze the data and train (bf16, from scratch; prints the adapter folder):

```powershell
brain-adapt export
```

Lens filter (use 1). One forward pass per train correction on the base model, no generation.
Drops the corrections the probe confidently disagrees with from LoRA training only; they stay in
the store and in retrieval. Read `lens_review.jsonl` afterwards:

```powershell
$env:QWEN_PATH = "C:\models\Qwen3.5-4B"
python adapt\train\lens_filter.py --export adapt\data\export\<head> --probe sidecar\probe_nothink.json --precision int8-t0
```

```powershell
python adapt\train\train_lora.py --export adapt\data\export\<head> --lens-filter adapt\data\export\<head>\lens_filter.json
```

3. Register and evaluate in the serving precision, with brain's probe and dev prompts as the canary:

```powershell
brain-adapt ledger register adapt\adapters\<adapter_id>
```

```powershell
cargo run --release -- dump-prompts --split probe --out runs\prompts_probe.jsonl
```

```powershell
python adapt\train\eval_adapt.py --export adapt\data\export\<head> --adapter adapt\adapters\<adapter_id> --precision int8-t0 --canary runs\prompts_probe.jsonl --canary runs\prompts_dev.jsonl --probe sidecar\probe_nothink.json --out adapt\evals\<adapter_id>
```

`--probe` adds the lens report (use 2) to `verdict.json` under `"lens"`, plus `lens.jsonl`. It never
changes pass or fail. How to read it:

| Field | Meaning |
|---|---|
| `heldout_adapter_vs_bare` | did the adapter move the internal judgment toward your corrections (wins, losses, mean shift in p(label)) |
| `said_vs_meant.said_not_meant` | text right, internal judgment still wrong: a surface fix that may not generalize |
| `said_vs_meant.meant_not_said` | judges right internally, says something else |
| `canary_internal_drift_kl` | how far the internal judgment moved on unrelated dev tasks |

4. Read `adapt\evals\<adapter_id>\review.jsonl`. If the verdict is `fail`, record that as the result
   (a null result, like the QRNG test) and stay on stage 1.

Expect about 4 x (heldout + 144) generations. On the laptop at about 12 tokens/s that's hours. On the 5080, much less.

## Stage 3: promote, watch, roll back

`--baseline-rate` is corrections per 100 interactions before this adapter:

```powershell
brain-adapt ledger promote <adapter_id> --verdict adapt\evals\<adapter_id>\verdict.json --baseline-rate 6.0 --reviewed
```

After real use, feed it the counts since promotion. It confirms the adapter, or rolls it back on its own:

```powershell
brain-adapt ledger check-live --interactions 120 --corrections 5
```

```powershell
brain-adapt ledger current
```

```powershell
brain-adapt ledger rollback --reason "felt worse"
```

The gate refuses: an unlocked or edited PREREG, a verdict from another adapter, base model,
dataset, precision or thinking setting, smoke runs, failing verdicts, and missing `--reviewed`.

**When the base model changes**, the current adapter is invalid. `brain-adapt ledger current --base-hash <hash>`
exits with code 2. Retrain from the same corrections. The corrections are the asset; adapters are build outputs.

## Design decisions (from the outside-view review)

| # | Failure | Decision here |
|---|---|---|
| F1 | 30 corrections only memorize | go/no-go at 150 with 30 gradable heldout |
| F2 | no baseline | adapter must beat retrieval, not the bare model |
| F3 | 24-item eval is noise | heldout corrections plus 144 canary prompts, paired exact test |
| F4 | model grades itself | deterministic checks only; human reads review.jsonl |
| F5 | train/eval leakage | split fixed by prompt hash at capture; heldout never retrievable |
| F6 | drift across rounds | always retrain from scratch; drift KL in the gate |
| F7 | base model changes | base weights hash bound into adapter, verdict and ledger |
| F8 | bad or poisoned corrections | conflict check, tombstones, local sources only |
| F9 | thinking-mode damage | thinking off only; no correct traces exist to train on |
| F10 | Windows / bitsandbytes trouble | train plain bf16 LoRA; int8 only for serving |
| F11 | gate passes, real use doesn't improve | provisional promotion, automatic rollback |
| F12 | abandonment | time box and kill criteria in PREREG_ADAPT.md |
| F13 | adapter trained without context, served with it (found by the learning check: half the gain lost) | training builds each prompt with leave-one-out retrieval context |
| F14 | corrections typed without the loop's system prompt train the wrong prompt shape | `--from-dump` capture; `add` and `stats` warn |
| F15 | store edited between export and train/eval | train and eval refuse a stale export |
| F16 | a fixed BM25 cutoff retrieved nothing while the store was small (found in the loop test) | a hit must cover `retrieval.min_coverage` (half) of the request's content words; BM25 only ranks |
| F17 | most real mistakes happen mid-run, but corrections only covered step 1 (found on "where is player-harness") | `brain correct --step N` keeps the conversation up to step N as `history`; training, eval and retrieval use it, and a test confirms the rebuilt prompt is byte-identical to what the loop sent |
| F19 | the lens as a training target would be learned around (the adapter moves the probe's readout, not the behavior) | the lens only picks rows (lens_filter.py) and reports (eval_adapt.py --probe); train_lora.py can't import lens code, checked by test_common.py; heldout is never filtered |
| F18 | word match only retrieved exact repeats (run 13: the LIMEN request never saw the player-harness lesson) | retrieval also compares meaning with a local embedding model; word match stays as the fallback; training and eval refuse to fall back |

## Not yet verified

- Nothing has run on Qwen3.5-4B or a GPU yet. The smoke test proves only that the plumbing works.
- `eval_adapt.py` reads brain's `dump-prompts` rows as either `messages` or `prompt`/`request`,
  with the label in `expected_first`. If the real file differs, it stops and prints the keys it found.
- The lens filter and lens report have only run with a fake probe on the tiny smoke model. The
  thinking-off probe doesn't exist yet (step 1b), and whether it holds up thinking-off is itself unknown.
- Loading LoRA through peft on Qwen3.5's Gated DeltaNet layers with `target_modules: all-linear`
  is untested. Run train_lora.py with `--smoke --max-steps 2` against the real model first.
