# Lens sidecar contract (v1)

The Rust loop already speaks this contract (`--gate lens` or `--shadow-lens`).
The sidecar itself is **not built yet on purpose**: it waits on the 8-bit spike
(does the J-lens readout on Qwen3.5-4B survive quantization?). If the spike fails,
the contract changes, so nothing gets written against it until then.

Default address: `http://127.0.0.1:8765`

## GET /health

Returns 200 when the model is loaded.

```json
{ "ok": true, "model": "Qwen/Qwen3.5-4B", "precision": "int8" }
```

## POST /gate

Request: the exact conversation the generator saw this step, plus the decision it proposed.

```json
{
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "What is 17 * 23?" }
  ],
  "proposed": "act"
}
```

Response: a distribution over the four decisions, read from internal state.

```json
{ "probs": { "act": 0.82, "answer": 0.10, "ask": 0.05, "abort": 0.03 } }
```

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
