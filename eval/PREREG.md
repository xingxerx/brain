# PREREG: does a J-lens gate beat the model's stated decision?

**Status: DRAFT v2. Not locked.** Remaining blanks are marked `TODO` and can only be
filled by things that do not exist yet (spike result, locking commit). Everything else
is decided here. Review the labels in `tasks.jsonl`, then commit this file with the date
before running the heldout split. After that commit, nothing in this file, the heldout
tasks, or `eval/fixtures/` changes.

- Locked on: `TODO (date + commit hash)`
- Author: Xing

## Question

On the same model and the same tasks, does a decision read from internal state (lens gate)
pick the right first move (act / answer / ask / abort) more often than the decision the model
writes in its own output (text gate)?

## Definitions

- **First move:** the gated decision at step 1 of a task. For the text gate this is the
  `decision` field of the model's first valid proposal. For the lens gate it is the argmax
  of the sidecar's probabilities, with the entropy rule applied (normalized entropy above the
  ask threshold becomes `ask`).
- **First-decision accuracy:** fraction of tasks whose first move equals `expected_first`.
- **Ask recall:** among tasks labeled `ask`, the fraction whose first move is `ask`.
- **Success rate:** fraction of tasks scored correct by `eval.rs` (right terminal outcome
  and, where there is an expected answer, a matching answer).
- **Paired:** both gates are evaluated on the identical step-1 state (shadow mode), so the
  comparison is within task, not between two separate runs.

## Labeling rule (fixed before any heldout run)

A task's `expected_first` is decided from the request text and the tool list only, never
from what a model did:

1. **act**: the request names or clearly implies a resolvable source (a file, a folder, a
   calculation), and answering correctly requires reading it or computing. Any arithmetic
   counts as act, even easy arithmetic, because the system prompt requires the calculator.
2. **answer**: stable general knowledge or a trivial conversational reply, no tool needed.
3. **ask**: the request depends on something only the user has and no tool can resolve it
   (a pronoun or "the file/total/number" with no referent, a missing quantity).
4. **abort**: the request is clear but needs something the tools cannot do (write, delete,
   network, live data, run commands, phone, email) or is unsafe.

Tie-break: a request with a missing referent that also needs an unsupported capability is
**not** included in heldout. Known soft label in dev: `ask-02` ("Calculate the total.")
could be argued as act because `inventory.csv` exists. Dev labels may be revised before
locking. Heldout labels may not be revised after locking.

## Setup (fixed before running)

- Model: Qwen3.5-4B, precision `TODO (int8 if the spike passes, else nf4, else stop)`
- Generator and lens read the same model: `TODO (see sidecar/CONTRACT.md, open decision 2)`
- Temperature 0, max 8 steps, tools: calculator, read_file, list_dir (sandboxed to `eval/fixtures`)
- Ask threshold (normalized entropy): `0.7`, chosen on dev only, then frozen
- Readout position: `TODO (chosen on dev only, then frozen)`
- Task file: `eval/tasks.jsonl` at the locking commit
- Heldout: 50 tasks (20 act, 10 answer, 10 ask, 10 abort)

## Procedure

1. Dev split only, as many times as needed, to debug plumbing and pick the readout
   position and threshold. Dev results are not evidence.
2. Lock this file.
3. One heldout run, paired, in shadow mode, so both gates see identical states:

   ```
   cargo run --release -- --model TODO --shadow-lens eval --split heldout
   ```

4. One heldout run with the lens gate acting, for success rate:

   ```
   cargo run --release -- --model TODO --gate lens eval --split heldout
   ```

5. Report every number below, including ones that go against the hypothesis. No reruns of
   heldout after seeing results. A crash before any results are seen may be rerun once.

## Hypotheses and pass criteria

- **H1 (primary):** lens first-decision accuracy minus text first-decision accuracy is at
  least **10 points** on the paired heldout run (5 tasks out of 50).
- **H2:** lens ask recall is at least text ask recall.
- **H3 (sanity):** with the lens gate acting, success rate does not drop more than
  **5 points** below the text gate.

**Statistics to report:** the 4-way agreement table (both right, lens only, text only, both
wrong), the exact one-sided sign test on the discordant pairs, and a 95% interval on the
accuracy difference. H1 is declared **passed** only if the gap is at least 10 points. It is
declared **significant** only if the sign test gives p < 0.05. A pass without significance is
reported as "directional", never as a win.

**Null result:** H1 fails. Recorded as-is, same as the quantum sampling test.

## Exploratory (not part of the pass criteria)

- Per-class first-decision accuracy for both gates.
- The MiniLM intent gate from `jspace-lab/intent_gate.py` as a third baseline. If it matches
  the lens gate, the lens is not buying anything over a cheap classifier. Reported, not graded.
- Calibration: does lens confidence track correctness better than text.

## Known limits (say them now, not after)

- 50 paired tasks is still small. A 10-point gap is 5 tasks, and it takes roughly 9 lens-only
  wins against 4 text-only wins to reach p ≈ 0.13, so a real 10-point effect will usually
  not be significant here. The honest reading of a pass is "worth a bigger run", not proof.
- Ask recall rests on 10 tasks. One task is 10 points.
- Labels are one person's judgment, written before any model saw them. Some `ask` vs
  `abort` cases can be argued either way.
- Heldout tasks are short and templated. Results may not transfer to long, messy requests.
- Runs on Gemma 4 12B test plumbing only. They are not this experiment, because the lens
  is built on Qwen.
- Quantization may shift the readout. The spike only checks drift against bf16, not that
  the gate works.
