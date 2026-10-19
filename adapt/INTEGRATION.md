# Wiring adapt into brain

Steps 1 to 3 are done (Oct 19 2026): `adapt` is a workspace member, `brain ask --adapt` puts
past corrections in front of the system prompt (`Agent::run_with_context`), and `brain correct <run>`
records a correction from a run log. A test against a fake model server confirmed that the loop's
prompt with context is byte-identical to what `train/common.py::messages_for` builds for training
and eval. Steps 4 and 5 are still open.

## 0. Ground rule: keep adapt out of the lens experiment

Retrieval changes the prompt, and an adapter changes the weights. brain's lens PREREG, the probe
and every `brain eval` run assume neither. So adapt is **off by default** and is only switched on
for interactive use (`brain ask`). Never mix the two in one run, and log which mode a run used.

## 1. Workspace

If the repo root has a `Cargo.toml` workspace, add `"adapt"` to `members`. Then in `loop/Cargo.toml`:

```toml
adapt = { path = "../adapt", package = "brain-adapt" }
```

## 2. Retrieval in the loop (stage 1)

In `agent.rs`, before step 1 builds its messages, behind a new `--adapt` flag that defaults to off:

```rust
let inf = adapt::for_request(std::path::Path::new(adapt::DEFAULT_DIR), &request, 3, 0.5)?;
if !inf.context.is_empty() {
    system_prompt = format!("{}\n\n{}", inf.context.trim_end(), system_prompt);
}
// log alongside the step so runs stay explainable:
// "adapt": {"hit_ids": inf.hit_ids, "adapter": inf.adapter.as_ref().map(|a| &a.adapter_id)}
```

The context goes **in front of** the system prompt, exactly as `train/common.py::messages_for` does.
Training, eval and serving must build the prompt the same way.

## 3. Capture from the loop

Add `brain correct` (or a key in `ask` mode) that calls `adapt::store::CorrectionStore::add` with:
- the exact `system` and user `prompt` the loop sent;
- `model_output`: the raw proposal;
- `source: "brain-user"`.

Never call `add` with text that came from a tool result, a file or the model. Only what you type.

## 4. Serving the adapter (stage 3, sidecar)

In `sidecar/engine.py`, after `load_model`:

```python
import json, subprocess, peft
cur = json.loads(subprocess.run(["adapt/target/release/brain-adapt", "ledger", "current", "--json"],
                                capture_output=True, text=True, check=True).stdout)
if cur and args.adapt:
    # refuse a stale adapter: it must match these exact base weights
    sys.path.insert(0, "adapt/train"); import common
    if cur["base_hash"] != common.base_hash(QWEN_PATH, "adapt/data/.base_hash_cache.json"):
        sys.exit("current adapter was trained on a different base model; retrain or roll back")
    hf = peft.PeftModel.from_pretrained(hf, cur["adapter_dir"])
```

**Caution:** the lens probe reads the late layers. An adapter changes them, so a probe trained on
the bare model is invalid with an adapter loaded. The sidecar should refuse `/gate` probe requests
while an adapter is active, the same way it already refuses a thinking-setting mismatch.

## 5. Counting for check-live

`brain-adapt ledger check-live` needs interactions and corrections since promotion. The simplest
source is the run logs: count `ask` steps, and count `add` events with `ts` after the promote event.
Add a `brain adapt-status` that does this count and calls `check-live`.
