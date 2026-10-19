"""J-lens readout for adapt: the sidecar's probe, applied offline.

Two uses, and only two (see PREREG_ADAPT.md, "The lens"):
  1. lens_filter.py  picks which train-split corrections the LoRA trains on.
  2. eval_adapt.py   reports whether the adapter moved the model's internal judgment.

The lens is NEVER a training signal. train_lora.py does not import this file, and
test_common.py checks that statically. If the lens became the thing the adapter is
optimized against, the adapter would learn to move the probe's readout instead of the
behavior, and the probe would stop meaning anything.

Readout matches the sidecar exactly: engine.Probe, last prompt token, layer L is
hidden_states[L + 1]. Only the rendering differs on purpose: adapt is thinking-off
only, so the probe must have been trained thinking-off too (probe.py collect --no-think).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "sidecar"))
import common  # noqa: E402
import engine  # noqa: E402  (numpy-only at import time; no model is loaded)
from common import die  # noqa: E402

# Why a row gets no lens reading. Rows outside the probe's training distribution are kept and
# counted, never silently scored: a readout off-distribution is not evidence of anything.
NA_NOT_DECISION = "not a decision correction (the probe only knows act/answer/ask/abort)"
NA_HISTORY = "later-step correction (the probe was trained on step-1 prompts only)"
NA_SYSTEM = "system prompt differs from the one the probe was trained on"


class Lens:
    def __init__(self, path, precision, smoke=False):
        self.path = path
        self.sha256 = common.sha256_file(path)
        self.probe = engine.Probe.load(path)
        t = self.probe.d["trained_on"]
        self.system_sha = t.get("system_prompt_sha256")
        self.smoke = smoke
        if t.get("enable_thinking") is not False:
            die(f"{path} was trained with enable_thinking={t.get('enable_thinking')}; adapt is thinking-off "
                "only, and a readout taken at a different token means nothing. Train a thinking-off probe:\n"
                "  python sidecar\\probe.py collect --no-think --prompts runs\\prompts_probe.jsonl "
                "--out runs\\acts_probe_nothink.npz\n"
                "  python sidecar\\probe.py train --acts runs\\acts_probe_nothink.npz "
                "--out sidecar\\probe_nothink.json")
        if t.get("precision") != precision:
            die(f"{path} was trained at precision {t.get('precision')} but this run uses {precision}; "
                "the readout must come from the same weights the probe was fit on")

    def summary(self):
        return {"path": os.path.abspath(self.path), "sha256": self.sha256, **self.probe.summary()}

    def na_reason(self, row):
        """None if the probe may read this row, else why not."""
        check = row.get("check") or {}
        if check.get("type") != "decision":
            return NA_NOT_DECISION
        if row.get("history"):
            return NA_HISTORY
        if not self.smoke and common.sha256_text(row.get("system") or "") != self.system_sha:
            return NA_SYSTEM
        return None

    def read(self, model, tok, msgs):
        """(probs dict, margin) at the last prompt token of the thinking-off render.
        Works on a plain model or a PeftModel (adapter on or disabled by the caller)."""
        import torch

        text = common.render(tok, msgs, "off")
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
        with torch.inference_mode():
            hs = model(**ids, output_hidden_states=True, use_cache=False).hidden_states
        n_blocks = len(hs) - 1
        states = {}
        for L in self.probe.layers:
            if not 0 <= L < n_blocks:
                die(f"probe layer {L} out of range 0..{n_blocks - 1} for this model")
            states[L] = hs[L + 1][0, -1].float().cpu().numpy()
        return self.probe.predict(states)


def top(probs):
    return max(probs, key=probs.get)


def kl(p, q, eps=1e-9):
    import math

    return sum(p[c] * (math.log(max(p[c], eps)) - math.log(max(q[c], eps))) for c in p)
