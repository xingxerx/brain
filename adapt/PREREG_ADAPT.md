# PREREG: does a LoRA adapter beat retrieval of past corrections?

Draft. The binding numbers are in `prereg.json`. This file explains them. Lock with
`brain-adapt prereg lock` **before** the first non-smoke eval. Any edit after locking makes
every later promotion fail until you re-lock and re-run, and a re-lock is recorded in the ledger.

## Question

On held-out corrections, does Qwen3.5-4B with a LoRA adapter trained on my corrections, **plus**
retrieval, do better than the same model with retrieval alone? And does it do that without
damaging behavior elsewhere?

## Conditions

All four use the same weights, greedy decoding, thinking off and the serving precision:
bare, retrieval, adapter, adapter + retrieval. Only adapter + retrieval vs retrieval is gated. The
rest are reported.

## Data

- Corrections: the train/heldout split is fixed at capture time by the hash of the normalized prompt
  (`heldout_frac` 0.2, pinned by the first correction). Heldout rows are never trained on and never retrievable.
- Canary: brain's `dump-prompts` rows from the **probe** and **dev** splits only. brain's own heldout
  belongs to its lens PREREG and is refused here.
- Only corrections with a deterministic check are scored.

## Go/no-go before stage 2

- At least `min_corrections` (150) live corrections and `min_heldout_gradable` (30) gradable heldout.
- Time box: **TODO fill in a date** (suggest 3 weeks from the first correction). If the go/no-go
  isn't met by then, stage 2 does not run. Record that as the result.

## Pass criteria (all required)

1. **Beats retrieval:** exact one-sided McNemar on heldout, adapter + retrieval vs retrieval,
   p < `gate.alpha` (0.05) **and** net wins >= `gate.min_net_wins` (3).
2. **No canary regression:** canary accuracy of adapter + retrieval >= retrieval minus
   `gate.max_canary_drop` (0.02).
3. **Bounded drift:** mean per-token KL(bare || adapter) over the first `gate.drift_tokens` (64) tokens
   of the bare model's own replies on `gate.drift_prompts` (50) canary prompts <= `gate.max_drift_kl` (0.05).
   *Disclosure:* 0.05 is a starting guess, not calibrated. If you want to calibrate it, do it on a
   smoke adapter **before** locking, and write down what you saw.
4. **Human review:** I read `review.jsonl` (a 10% sample plus every canary regression) and promote with `--reviewed`.

## If it fails

Stay on stage 1. Record the verdict file and the numbers. Allowed follow-ups, each needing a
new lock: more corrections (same thresholds), or one change to `lora` (rank or epochs). No
threshold changes after seeing results.

## Live rollback (stage 3)

Promotion is provisional. After `live.min_interactions` (100) interactions, the adapter is confirmed
if corrections per 100 interactions <= baseline x (1 + `live.tolerance`), using a floor of 1 per 100
for the baseline. It rolls back automatically, as early as 20 interactions, once there are at least
3 corrections and the rate is over that limit.

## Known limits

- Thinking off only (no correct reasoning traces exist to train on).
- One user's corrections: the result says nothing about anyone else's.
- 30 to 60 heldout items gives low power. A real but small gain can fail criterion 1. That's an
  accepted cost of not fooling myself.
