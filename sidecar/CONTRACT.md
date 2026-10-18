# Lens sidecar contract (v1)

The Rust loop already speaks this contract (`--gate lens` or `--shadow-lens`).
The sidecar is `server.py` (v1; shared code in `engine.py`). The 8-bit spike passed on the late band (L20-30), so it
serves int8 Qwen3.5-4B. The default is `int8-t0` (no outlier decomposition, ~2x faster decode),
which also passed, narrowly; `PRECISION=int8` gives bitsandbytes' default threshold. See
`jspace-lab/readout/SPIKE_PREREG.md` for the criterion and both results.
`python sidecar\server.py --fake` runs it with canned replies, no model, to test wiring.

Default address: `http://127.0.0.1:8765`

## GET /health

Returns 200 when the model is loaded.

```json
{ "ok": true, "model": "Qwen/Qwen3.5-4B", "precision": "int8-t0", "gate_mode": "probe",
  "fast_path": { "flash_linear_attention": false, "causal_conv1d": false },
  "probe": { "layers": [20, 21, 22], "lambda": 1.0, "cv_accuracy": 0.9, "enable_thinking": true } }
```

## POST /gate

Request: the exact conversation the generator saw this step, plus the decision it proposed.

```json
{
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "What is 17 * 23?" }
  ],
  "proposed": "act",
  "chat_template_kwargs": { "enable_thinking": false }
}
```

`chat_template_kwargs` is optional and must match what the generator was sent (the Rust side
sends it with `--no-think`). The gate renders the prompt with it, so both read the same text.

Response: a distribution over the four decisions, read from internal state, plus whatever the
mode reports. The Rust side logs every extra key under `gate.meta` in `steps.jsonl`.

```json
{ "probs": { "act": 0.82, "answer": 0.10, "ask": 0.05, "abort": 0.03 },
  "mode": "probe", "layers": [20, 21, 22], "margin": 3.1, "thinking": true, "ms": 180.0 }
```

```json
{ "probs": { "act": 0.82, "answer": 0.10, "ask": 0.05, "abort": 0.03 },
  "mode": "anchors", "anchor_mass": 0.004, "thinking": true, "ms": 210.0 }
```

Errors: 400 with `{"error": "..."}` for a request the gate must refuse (probe trained with a
different thinking setting, prompt longer than `GATE_MAX_TOKENS`), 500 for anything else. The
Rust side shows the body.

## Gate modes

- **probe** (v1, `--probe sidecar\probe.json`): multinomial logistic regression on the
  last-prompt-token state of the chosen layer or band, trained by `sidecar/probe.py` on the probe
  split only. The probe file records precision, thinking setting and system-prompt hash; the
  server refuses a mismatched precision at startup and a mismatched thinking setting per request.
- **anchors** (v0, default without `--probe`): J-lens logits on L20-30 scored against anchor words.
  Dev, 2026-10-18 run 6: 6/24 with the gate acting, against 24/24 for the text gate.

In anchors mode, `anchor_mass` is the total lens probability that landed on the anchor words before normalizing.
If it is tiny, the four probs are normalized noise. Add `"debug": true` to the request to get
the raw per-class mass and the top-10 lens tokens of the last late layer.

Rules the Rust side applies (see `loop/src/gate.rs`):

- Missing keys count as 0. Values don't need to sum to 1; they get normalized.
- Normalized entropy above `--ask-threshold` (default 0.7) turns the decision into `ask`,
  the same rule as `jspace-lab/intent_gate.py`.
- Otherwise the top class wins. If it differs from `proposed`, the step is logged as an override.

## Open decisions for the spike to settle

1. **Readout position.** Last prompt token before generation (decides before the model writes
   anything), or the token where the model writes its `decision` field (reads what it is about to say).
   The first is the more interesting claim.
2. **Same model for generating and reading.** The gate is only meaningful when it reads the model
   that is acting. Options: run Qwen3.5-4B in LM Studio as the generator (GGUF weights, close but not
   identical to the sidecar's), or have the sidecar also serve `/v1/chat/completions` so one process
   does both (identical weights, simpler claim). The second is cleaner.
3. **Chat template.** The sidecar must apply the model's own chat template to `messages`,
   matching what the generator saw.

## Chat completions

Standard OpenAI shape. `chat_template_kwargs.enable_thinking=false` renders the template with
an empty, closed think block; then the whole reply is `content`. `usage.timing` carries
`ttft_ms`, `total_ms` and `decode_tok_s`, and the server prints the same per request.

## Decided in v0

- Decision 1 (readout position): last prompt token, before generation, averaged over L20-30.
- Decision 2 (same model): the sidecar also serves `/v1/chat/completions`, so one process,
  one set of weights, does both. Run the loop with
  `--base-url http://127.0.0.1:8765/v1 --model Qwen/Qwen3.5-4B --no-schema`.
- Decision 3 (template): the model's own chat template, thinking left on (its default). The
  thinking block is returned in `message.reasoning_content`; `message.content` holds only the
  text after `</think>`. If thinking never closes (max_tokens hit), content is empty and
  finish_reason is "length".
