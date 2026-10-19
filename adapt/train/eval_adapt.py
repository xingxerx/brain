"""Stage 2 eval: does the adapter beat retrieval, without breaking anything?

    python adapt/train/eval_adapt.py --export adapt/data/export/<id> --adapter adapt/adapters/<adapter_id> ^
        --base C:\\models\\Qwen3.5-4B --precision int8-t0 ^
        --canary runs\\prompts_probe.jsonl --canary runs\\prompts_dev.jsonl --out adapt/evals/<adapter_id>

Four conditions, same model load, greedy, thinking off:
  bare               no context, adapter disabled
  retrieval          past corrections in context (stage 1, the baseline to beat)
  adapter            adapter, no context (reported, not gated)
  adapter_retrieval  adapter plus context (what would be served)

Items:
  heldout corrections  every heldout row with a check; graded by that check only (F4)
  canary               brain's dump-prompts rows (probe and dev splits ONLY; heldout is refused,
                       it belongs to brain's own PREREG); graded on the first decision

Pass needs all of (thresholds come from the locked PREREG):
  1. adapter_retrieval beats retrieval on heldout corrections: exact one-sided McNemar
     p < gate.alpha AND net wins >= gate.min_net_wins
  2. canary accuracy of adapter_retrieval >= retrieval - gate.max_canary_drop
  3. mean per-token KL(bare || adapter) on gate.drift_prompts canary prompts <= gate.max_drift_kl
  4. the adapter matches this base model, dataset and PREREG
The verdict file is input to `brain-adapt ledger promote`, which re-checks the bindings.
A human still reads review.jsonl before promoting.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import die  # noqa: E402

CONDITIONS = ("bare", "retrieval", "adapter", "adapter_retrieval")
LABEL_KEYS = ("expected_first", "expected", "label", "decision")


def mcnemar_one_sided(b, c):
    """P(X >= b) for X ~ Binomial(b + c, 0.5). b = new wins, c = new losses."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def canary_rows(paths):
    """brain's dump-prompts output. Accepts messages=[...] or prompt/request (+system)."""
    rows = []
    for p in paths:
        for i, r in enumerate(common.read_jsonl(p)):
            split = r.get("split")
            if split == "heldout":
                die(f"{p}: canary contains brain heldout rows; those belong to brain's PREREG, not this eval")
            label = next((r[k] for k in LABEL_KEYS if r.get(k) in common.DECISIONS), None)
            if label is None:
                die(f"{p} row {i + 1}: no decision label in any of {LABEL_KEYS}; keys are {sorted(r)}")
            if isinstance(r.get("messages"), list) and r["messages"]:
                msgs = r["messages"]
                system = "\n\n".join(m["content"] for m in msgs if m.get("role") == "system") or None
                users = [m["content"] for m in msgs if m.get("role") == "user"]
                if len(users) != 1 or len(msgs) - (1 if system else 0) != 1:
                    die(f"{p} row {i + 1}: expected one system + one user message (step-1 prompts)")
                prompt = users[0]
            else:
                prompt = r.get("prompt") or r.get("request")
                system = r.get("system")
                if not prompt:
                    die(f"{p} row {i + 1}: no messages, prompt or request; keys are {sorted(r)}")
            rows.append({"id": r.get("task_id") or r.get("id") or f"{os.path.basename(p)}:{i + 1}",
                         "prompt": prompt, "system": system, "label": label, "split": split})
    ids = [r["id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        die(f"duplicate canary ids across --canary files: {dup[:5]}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--adapter", help="adapter folder; omit for a retrieval-only report (stage 1)")
    ap.add_argument("--base", default=os.environ.get("QWEN_PATH"))
    ap.add_argument("--precision", required=True, help="must equal the PREREG serving_precision")
    ap.add_argument("--canary", action="append", default=[], help="dump-prompts JSONL (repeatable)")
    ap.add_argument("--out", required=True, help="folder for verdict.json, outputs.jsonl, review.jsonl")
    ap.add_argument("--prereg", default="adapt/prereg.json")
    ap.add_argument("--dir", default="adapt/data")
    ap.add_argument("--adapt-bin")
    ap.add_argument("--smoke", action="store_true", help="plumbing test; the verdict can never be promoted")
    ap.add_argument("--limit", type=int, default=0, help="smoke only: cap items per set")
    a = ap.parse_args()

    import torch

    if not a.base:
        die("pass --base or set QWEN_PATH")
    pr, pr_sha = common.load_prereg(a.prereg)
    g = pr["gate"]
    binary = common.adapt_bin(a.adapt_bin)
    if not a.smoke:
        common.require_locked(binary, a.dir, a.prereg)
        if a.precision != pr["serving_precision"]:
            die(f"--precision {a.precision} but the PREREG serving_precision is {pr['serving_precision']}: "
                "eval must run exactly as the loop serves")
        if not a.canary:
            die("at least one --canary file is required")
        if a.limit:
            die("--limit is for smoke runs only")
    thinking = pr["thinking"]
    if thinking != "off":
        die("only thinking=off is supported (see train_lora.py, F9)")

    with open(os.path.join(a.export, "manifest.json"), encoding="utf-8") as f:
        em = json.load(f)
    common.require_export_current(a.dir, em)
    held = [r for r in common.read_jsonl(os.path.join(a.export, "heldout.jsonl"))
            if r["check"]["type"] != "none"]
    if any(r["split"] != "heldout" for r in held):
        die("heldout.jsonl contains non-heldout rows")
    canary = canary_rows(a.canary)
    if a.limit:
        held, canary = held[:a.limit], canary[:a.limit]
    if not held:
        die("no gradable heldout corrections")

    print("hashing base weights (cached after the first run)...", flush=True)
    bh = common.base_hash(a.base, os.path.join(a.dir, ".base_hash_cache.json"))
    am = None
    reasons = []
    if a.adapter:
        with open(os.path.join(a.adapter, "manifest.json"), encoding="utf-8") as f:
            am = json.load(f)
        # F7: the adapter must belong to exactly this base, dataset and PREREG.
        for k, want in (("base_hash", bh), ("data_sha256", em["data_sha256"]), ("prereg_sha256", pr_sha)):
            if am.get(k) != want:
                msg = f"adapter {k} {str(am.get(k))[:12]} != {str(want)[:12]}"
                if a.smoke:
                    reasons.append(msg)
                else:
                    die(msg + " (retrain on this base/data/PREREG)")

    tok, model = common.load_model(a.base, a.precision, smoke=a.smoke)
    if a.adapter:
        import peft
        model = peft.PeftModel.from_pretrained(model, a.adapter)
        model.eval()
    dev = next(model.parameters()).device
    retr = common.Retriever(binary, a.dir, a.prereg, pr["retrieval"]["k"], pr["retrieval"]["min_coverage"])
    max_new = pr["generation"]["max_new_tokens"]

    def adapter_ctx(on):
        import contextlib
        if not a.adapter or on:
            return contextlib.nullcontext()
        return model.disable_adapter()

    @torch.no_grad()
    def generate(msgs):
        text = common.render(tok, msgs, thinking)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        new = out[0, ids["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True), new.tolist()

    conds = CONDITIONS if a.adapter else ("bare", "retrieval")
    items = [("heldout", r) for r in held] + [("canary", r) for r in canary]
    results = {c: {} for c in conds}
    bare_ids = {}
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    n_total, n_done = len(items) * len(conds), 0
    with open(os.path.join(a.out, "outputs.jsonl"), "w", encoding="utf-8") as fo:
        for cond in conds:
            use_adapter = cond.startswith("adapter")
            use_ctx = cond.endswith("retrieval")
            with adapter_ctx(use_adapter):
                for kind, r in items:
                    hits = retr(r["prompt"]) if use_ctx else {"context": "", "hits": []}
                    text, ids = generate(common.messages_for(r["prompt"], r.get("system"), hits["context"]))
                    if kind == "heldout":
                        ok = common.passes(r["check"], text, r["correction"])
                    else:
                        ok = common.extract_decision(text) == r["label"]
                    results[cond][(kind, r["id"])] = ok
                    if cond == "bare" and kind == "canary":
                        bare_ids[r["id"]] = ids
                    fo.write(json.dumps({"condition": cond, "set": kind, "id": r["id"], "pass": ok,
                                         "hit_ids": [h["id"] for h in hits["hits"]], "output": text},
                                        ensure_ascii=False) + "\n")
                    n_done += 1
                    if n_done % 10 == 0 or n_done == n_total:
                        el = time.time() - t0
                        print(f"{n_done}/{n_total} generations, {el:.0f}s, eta {el / n_done * (n_total - n_done):.0f}s",
                              flush=True)

    def acc(cond, kind):
        v = [ok for (k, _), ok in results[cond].items() if k == kind]
        return (sum(v) / len(v)) if v else None

    summary = {c: {"heldout": acc(c, "heldout"), "canary": acc(c, "canary")} for c in conds}

    def paired(new, old, kind):
        keys = [k for k in results[new] if k[0] == kind]
        b = sum(results[new][k] and not results[old][k] for k in keys)
        c = sum(results[old][k] and not results[new][k] for k in keys)
        return {"new_wins": b, "new_losses": c, "p_one_sided": round(mcnemar_one_sided(b, c), 6)}

    comparisons = {"retrieval_vs_bare_heldout": paired("retrieval", "bare", "heldout")}
    drift = None
    if a.adapter:
        comparisons["adapter_retrieval_vs_retrieval_heldout"] = paired("adapter_retrieval", "retrieval", "heldout")
        comparisons["adapter_retrieval_vs_retrieval_canary"] = paired("adapter_retrieval", "retrieval", "canary")

        # Drift: teacher-force the bare model's own greedy reply and compare the
        # next-token distributions with and without the adapter (no context).
        kls = []
        dp = canary[:g["drift_prompts"]]
        with torch.no_grad():
            for r in dp:
                text = common.render(tok, common.messages_for(r["prompt"], r.get("system")), thinking)
                p_ids = tok(text, add_special_tokens=False)["input_ids"]
                o_ids = bare_ids.get(r["id"], [])[:g["drift_tokens"]]
                x = torch.tensor([p_ids + o_ids], device=dev)
                sl = slice(len(p_ids) - 1, len(p_ids) - 1 + max(1, len(o_ids)))
                with model.disable_adapter():
                    lb = torch.log_softmax(model(input_ids=x).logits[0, sl].float(), -1)
                la = torch.log_softmax(model(input_ids=x).logits[0, sl].float(), -1)
                kls.append((lb.exp() * (lb - la)).sum(-1).mean().item())
        drift = {"prompts": len(kls), "mean_kl": round(sum(kls) / len(kls), 6) if kls else None,
                 "max_kl": round(max(kls), 6) if kls else None}

        m = comparisons["adapter_retrieval_vs_retrieval_heldout"]
        if not (m["p_one_sided"] < g["alpha"] and m["new_wins"] - m["new_losses"] >= g["min_net_wins"]):
            reasons.append(f"adapter does not beat retrieval on heldout: {m['new_wins']} wins, "
                           f"{m['new_losses']} losses, p={m['p_one_sided']} (need p<{g['alpha']} and net>={g['min_net_wins']})")
        ca, cr = summary["adapter_retrieval"]["canary"], summary["retrieval"]["canary"]
        if ca is None:
            reasons.append("no canary items")
        elif ca < cr - g["max_canary_drop"]:
            reasons.append(f"canary regressed: {ca:.3f} vs retrieval {cr:.3f} (max drop {g['max_canary_drop']})")
        if drift["mean_kl"] is None or drift["mean_kl"] > g["max_drift_kl"]:
            reasons.append(f"drift too high: mean KL {drift['mean_kl']} > {g['max_drift_kl']}")
        if a.smoke:
            reasons.append("smoke run")

    # Review file: a seeded sample of heldout items plus every canary regression.
    rng = random.Random(pr["lora"]["seed"])
    k = max(1, math.ceil(g["review_fraction"] * len(held)))
    sample = {r["id"] for r in rng.sample(held, min(k, len(held)))}
    outs = {}
    with open(os.path.join(a.out, "outputs.jsonl"), encoding="utf-8") as f:
        for l in f:
            o = json.loads(l)
            outs.setdefault((o["set"], o["id"]), {})[o["condition"]] = o
    with open(os.path.join(a.out, "review.jsonl"), "w", encoding="utf-8") as f:
        top = "adapter_retrieval" if a.adapter else "retrieval"
        for r in held:
            if r["id"] in sample:
                f.write(json.dumps({"why": "random sample", "id": r["id"], "prompt": r["prompt"],
                                    "correction": r["correction"], "check": r["check"],
                                    "outputs": {c: outs[("heldout", r["id"])][c] for c in conds}},
                                   ensure_ascii=False) + "\n")
        for r in canary:
            o = outs[("canary", r["id"])]
            if o["retrieval"]["pass"] and not o[top]["pass"]:
                f.write(json.dumps({"why": "canary regression", "id": r["id"], "prompt": r["prompt"],
                                    "label": r["label"], "outputs": o}, ensure_ascii=False) + "\n")

    verdict = {
        "adapter_id": am["adapter_id"] if am else None,
        "adapter_dir": os.path.abspath(a.adapter) if a.adapter else None,
        "base_hash": bh,
        "data_sha256": em["data_sha256"],
        "prereg_sha256": pr_sha,
        "precision": a.precision,
        "thinking": thinking,
        "smoke": bool(a.smoke),
        "n_heldout": len(held),
        "n_canary": len(canary),
        "summary": summary,
        "comparisons": comparisons,
        "drift": drift,
        "verdict": ("pass" if not reasons else "fail") if a.adapter else "n/a (no adapter)",
        "reasons": reasons,
        "review_file": os.path.abspath(os.path.join(a.out, "review.jsonl")),
        "seconds": round(time.time() - t0, 1),
    }
    common.write_json(os.path.join(a.out, "verdict.json"), verdict)
    print(json.dumps({k: verdict[k] for k in ("summary", "comparisons", "drift", "verdict", "reasons")}, indent=2))
    print(f"\nverdict -> {os.path.join(a.out, 'verdict.json')}")


if __name__ == "__main__":
    main()
