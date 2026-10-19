"""Use 1 of the lens: choose which train-split corrections the LoRA trains on.

    python adapt/train/lens_filter.py --export adapt/data/export/<head> --probe sidecar/probe_nothink.json ^
        --base C:\\models\\Qwen3.5-4B --precision int8-t0

One forward pass per correction, no generation, on the BASE model (no adapter, no
retrieval context: the prompt shape the probe was trained on). For each decision
correction the probe says which first move the model internally judges right:

  agree        lens top class == the correction's decision. The model already "knew";
               only its words were wrong. Cheapest fix to train in. Kept.
  uncertain    lens disagrees, but below lens.drop_confidence. Kept.
  contradicts  lens confidently picks a different decision. Either the correction is
               a mistake, or it asks the model to go against a strong internal judgment
               (most drift for least gain). Dropped from LoRA training only.

Dropped rows stay in the correction store and in retrieval (stage 1 still uses them).
Nothing is deleted. Read lens_review.jsonl: if a dropped correction is right, keep it
in the store and accept that this filter cost you one training row.

Rules that keep the result honest:
  - Train split only. Heldout is never filtered: the eval must score every heldout row,
    or the filter could make the adapter look better by choosing its own test.
  - Thresholds come from the locked PREREG (lens.*). No tuning after seeing results.
  - If more than lens.max_drop_frac of the train split would be dropped, stop: that means
    the probe is off-distribution, not that your corrections are bad.
  - The probe must be thinking-off, at the serving precision, and the one the PREREG pins.
  - Rows the probe can't read (non-decision, later-step, other system prompt) are kept and
    counted as n/a.

Writes <export>/lens_filter.json (input to train_lora.py --lens-filter) and
<export>/lens_review.jsonl (every dropped row, for you to read).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import lens as lenslib  # noqa: E402
from common import die  # noqa: E402


def classify(probs, label, drop_confidence):
    t = lenslib.top(probs)
    if t == label:
        return "agree"
    if probs[t] >= drop_confidence:
        return "contradicts"
    return "uncertain"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--probe", required=True, help="thinking-off probe JSON from sidecar/probe.py train")
    ap.add_argument("--base", default=os.environ.get("QWEN_PATH"))
    ap.add_argument("--precision", required=True, help="must equal the probe's and the PREREG serving precision")
    ap.add_argument("--prereg", default="adapt/prereg.json")
    ap.add_argument("--dir", default="adapt/data")
    ap.add_argument("--adapt-bin")
    ap.add_argument("--smoke", action="store_true", help="plumbing test; skips lock and pin checks")
    a = ap.parse_args()

    if not a.base:
        die("pass --base or set QWEN_PATH")
    pr, pr_sha = common.load_prereg(a.prereg)
    lc = pr.get("lens") or die("prereg.json has no \"lens\" section")
    if not a.smoke:
        common.require_locked(common.adapt_bin(a.adapt_bin), a.dir, a.prereg)
        if not lc.get("filter"):
            die("the PREREG says lens.filter is false; the filter is not part of this experiment")
        if a.precision != pr["serving_precision"]:
            die(f"--precision {a.precision} != PREREG serving_precision {pr['serving_precision']}")
    lens = lenslib.Lens(a.probe, a.precision, smoke=a.smoke)
    if not a.smoke and lens.sha256 != lc.get("probe_sha256"):
        die(f"probe sha256 {lens.sha256[:12]} is not the one the PREREG pins ({str(lc.get('probe_sha256'))[:12]})")

    with open(os.path.join(a.export, "manifest.json"), encoding="utf-8") as f:
        em = json.load(f)
    common.require_export_current(a.dir, em)
    train = common.read_jsonl(os.path.join(a.export, "train.jsonl"))
    if any(r["split"] != "train" for r in train):
        die("train.jsonl contains non-train rows; the filter never touches heldout")
    if not train:
        die("empty train split")

    settings = {"drop_confidence": lc["drop_confidence"], "max_drop_frac": lc["max_drop_frac"],
                "precision": a.precision, "thinking": "off", "context": "none", "model": "base"}
    tok, model = common.load_model(a.base, a.precision, smoke=a.smoke)
    t0 = time.time()
    rows, review = [], []
    for i, r in enumerate(train):
        label = (r.get("check") or {}).get("value")
        said = common.extract_decision(r.get("model_output", ""))
        na = lens.na_reason(r)
        if na:
            rows.append({"id": r["id"], "status": "n/a", "why": na})
            continue
        probs, margin = lens.read(model, tok, common.messages_for(r["prompt"], r.get("system")))
        status = classify(probs, label, lc["drop_confidence"])
        row = {"id": r["id"], "status": status, "label": label, "said": said, "lens_top": lenslib.top(probs),
               "p_label": round(probs[label], 6), "margin": round(margin, 4),
               "probs": {k: round(v, 6) for k, v in probs.items()}}
        rows.append(row)
        if status == "contradicts":
            review.append({**row, "prompt": r["prompt"], "model_output": r.get("model_output"),
                           "correction": r["correction"]})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(train)}  {time.time() - t0:.0f}s", flush=True)

    counts = {s: sum(x["status"] == s for x in rows) for s in ("agree", "uncertain", "contradicts", "n/a")}
    dropped = sorted(x["id"] for x in rows if x["status"] == "contradicts")
    scored = [x for x in rows if x["status"] != "n/a"]
    # "Knew but said wrong": the lens already holds the corrected decision and the
    # model's own reply said something else. These are the corrections most likely
    # to train in cleanly, and the count is a direct read of the said-vs-meant gap.
    knew = sum(x["status"] == "agree" and x["said"] not in (None, x["label"]) for x in scored)
    body = {
        "export_chain_head": em["chain_head"],
        "train_sha256": em["train_sha256"],
        "data_sha256": em["data_sha256"],
        "probe_sha256": lens.sha256,
        "prereg_sha256": pr_sha,
        "settings": settings,
        "smoke": bool(a.smoke),
        "probe": lens.summary(),
        "n_train": len(train),
        "counts": counts,
        "knew_but_said_wrong": knew,
        "dropped": dropped,
        "rows": rows,
        "seconds": round(time.time() - t0, 1),
    }
    body["filter_sha256"] = common.lens_filter_sha(body)
    frac = len(dropped) / len(train)
    body["drop_frac"] = round(frac, 4)
    common.write_json(os.path.join(a.export, "lens_filter.json"), body)
    with open(os.path.join(a.export, "lens_review.jsonl"), "w", encoding="utf-8") as f:
        for x in review:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(json.dumps({"counts": counts, "knew_but_said_wrong": knew, "drop_frac": body["drop_frac"]}, indent=2))
    if frac > lc["max_drop_frac"]:
        msg = (f"the filter would drop {len(dropped)}/{len(train)} ({frac:.0%}) > max_drop_frac "
               f"{lc['max_drop_frac']:.0%}. That points at the probe (off-distribution), not the data. "
               "Record it as a result; don't raise the threshold.")
        if not a.smoke:
            os.replace(os.path.join(a.export, "lens_filter.json"), os.path.join(a.export, "lens_filter.REFUSED.json"))
            die(msg)
        print("smoke: " + msg)
    print(f"\nfilter {body['filter_sha256'][:12]} -> {os.path.join(a.export, 'lens_filter.json')}")
    print(f"read the {len(review)} dropped rows: {os.path.join(a.export, 'lens_review.jsonl')}")
    print("next: train_lora.py ... --lens-filter " + os.path.join(a.export, "lens_filter.json"))


if __name__ == "__main__":
    main()
