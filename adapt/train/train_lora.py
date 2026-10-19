"""Train one LoRA adapter from scratch on the exported train split.

    python adapt/train/train_lora.py --export adapt/data/export/<id> --base C:\\models\\Qwen3.5-4B

Design (from the outside-view review, see adapt/README.md):
  F6  Always from scratch on the full live set. No stacking, no replay buffer:
      the whole set is the replay. Deleting a correction removes it next time.
  F7  The adapter records the exact base weights hash, the dataset hash and the
      locked PREREG hash. The ledger refuses a verdict that does not match.
  F9  Thinking off only. There are no correct reasoning traces for corrections,
      and training a final answer after the model's own wrong trace teaches it
      to contradict itself. Serving must match (brain --no-think).
  F10 Plain bf16 LoRA, no QLoRA: a 4B model in bf16 is about 8 GB and fits a
      16 GB card with gradient checkpointing. No bitsandbytes needed to train.
  Loss is on the assistant reply only; prompt tokens are masked with -100.
  Rows that do not fit max_len are skipped and reported, never truncated.
  Lens: --lens-filter only removes rows before training (lens_filter.py decided which).
      This file never loads the probe or reads hidden states, so the lens can't leak
      into the loss. test_common.py checks that.
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


def build_example(tok, row, max_len, context=""):
    """Returns (input_ids, labels) or a reason string if the row can't be used.
    `context` is the leave-one-out retrieval block, so training sees the same
    prompt shape the adapter is served with (adapter + retrieval)."""
    msgs = common.messages_for(row["prompt"], row.get("system"), context, row.get("history"))
    prompt_text = common.render(tok, msgs, "off")
    full_text = tok.apply_chat_template(msgs + [{"role": "assistant", "content": row["correction"]}],
                                        tokenize=False, enable_thinking=False)
    if not full_text.startswith(prompt_text):
        # Some templates render a finished assistant turn differently from the
        # generation prompt. Fall back to the generation prompt + reply + EOS,
        # which is exactly what the model sees at inference time.
        if tok.eos_token is None:
            return "template renders the assistant turn differently and the tokenizer has no eos_token"
        full_text = prompt_text + row["correction"] + tok.eos_token
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    if f_ids[:len(p_ids)] != p_ids:
        return "prompt tokens are not a prefix of the full tokens (boundary merge)"
    if len(f_ids) > max_len:
        return f"too long ({len(f_ids)} > max_len {max_len})"
    if len(f_ids) == len(p_ids):
        return "empty target"
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return f_ids, labels


def load_lens_filter(a, pr, pr_sha, em):
    """The filter must belong to this exact export and PREREG, and the PREREG decides
    whether one is used at all (a pre-registered choice, not a per-run one)."""
    want = bool((pr.get("lens") or {}).get("filter"))
    if not a.lens_filter:
        if want and not a.smoke:
            die("the PREREG says lens.filter is true: run lens_filter.py and pass --lens-filter")
        return None
    if not want and not a.smoke:
        die("--lens-filter given but the PREREG says lens.filter is false")
    with open(a.lens_filter, encoding="utf-8") as f:
        lf = json.load(f)
    if common.lens_filter_sha(lf) != lf.get("filter_sha256"):
        die(f"{a.lens_filter} was edited after lens_filter.py wrote it")
    for k, want_v in (("export_chain_head", em["chain_head"]), ("train_sha256", em["train_sha256"])):
        if lf.get(k) != want_v:
            die(f"lens filter {k} does not match this export; re-run lens_filter.py on it")
    if not a.smoke:
        if lf.get("smoke"):
            die("that lens filter was a smoke run")
        if lf.get("prereg_sha256") != pr_sha:
            die("lens filter was made under a different PREREG")
        if lf.get("probe_sha256") != pr["lens"].get("probe_sha256"):
            die("lens filter used a probe the PREREG does not pin")
    return lf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="folder from `brain-adapt export`")
    ap.add_argument("--base", default=os.environ.get("QWEN_PATH"), help="local model folder (or QWEN_PATH)")
    ap.add_argument("--out", default="adapt/adapters")
    ap.add_argument("--prereg", default="adapt/prereg.json")
    ap.add_argument("--dir", default="adapt/data", help="brain-adapt data folder")
    ap.add_argument("--adapt-bin")
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing test: skips the lock and size checks, fp32 allowed, never promotable")
    ap.add_argument("--precision", default="bf16", help="training precision (bf16; fp32 only with --smoke)")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only: stop after this many optimizer steps")
    ap.add_argument("--lens-filter", help="lens_filter.json from lens_filter.py; required when the PREREG says lens.filter")
    a = ap.parse_args()

    import torch
    import peft

    if not a.base:
        die("pass --base or set QWEN_PATH")
    pr, pr_sha = common.load_prereg(a.prereg)
    if pr.get("thinking") != "off":
        die("thinking must be \"off\" in the PREREG: there are no correct reasoning traces to train on (F9)")
    if a.max_steps and not a.smoke:
        die("--max-steps is for smoke runs only")
    if a.precision != "bf16" and not a.smoke:
        die("real runs train in bf16")
    binary = common.adapt_bin(a.adapt_bin)
    if not a.smoke:
        common.require_locked(binary, a.dir, a.prereg)

    with open(os.path.join(a.export, "manifest.json"), encoding="utf-8") as f:
        em = json.load(f)
    common.require_export_current(a.dir, em)
    train = common.read_jsonl(os.path.join(a.export, "train.jsonl"))
    total = em["n_train"] + em["n_heldout"]
    if not a.smoke:
        if total < pr["min_corrections"]:
            die(f"{total} corrections < min_corrections {pr['min_corrections']}: stage 2 is not a go yet")
        if em["n_heldout_gradable"] < pr["min_heldout_gradable"]:
            die(f"{em['n_heldout_gradable']} gradable heldout < {pr['min_heldout_gradable']}")
    if any(r["split"] != "train" for r in train):
        die("train.jsonl contains non-train rows; refusing (leakage)")
    lens_filter = load_lens_filter(a, pr, pr_sha, em)
    if lens_filter:
        drop = set(lens_filter["dropped"])
        train = [r for r in train if r["id"] not in drop]
        print(f"lens filter {lens_filter['filter_sha256'][:12]}: dropped {len(drop)}, training on {len(train)}")
        if not train:
            die("the lens filter dropped every training row")

    lc = pr["lora"]
    cache = os.path.join(a.dir, ".base_hash_cache.json")
    print("hashing base weights (cached after the first run)...", flush=True)
    bh = common.base_hash(a.base, cache)
    config = {k: lc[k] for k in sorted(lc)}
    id_parts = [bh, em["data_sha256"], config, pr_sha, a.precision, "smoke" if a.smoke else "real"]
    if lens_filter:  # a filtered and an unfiltered adapter on the same data are different adapters
        id_parts.append(lens_filter["filter_sha256"])
    adapter_id = "a_" + common.sha256_text(common.canonical(id_parts))[:12]
    out_dir = os.path.join(a.out, adapter_id)
    if os.path.exists(os.path.join(out_dir, "manifest.json")):
        die(f"{out_dir} already exists: same base, data, config and PREREG were already trained")

    random.seed(lc["seed"])
    torch.manual_seed(lc["seed"])
    tok, model = common.load_model(a.base, a.precision, smoke=a.smoke)

    # Train on the served shape: retrieval context in front, leave-one-out so a
    # row never sees its own correction. Found by the CPU learning check: an
    # adapter trained without context lost half its gain when served with it.
    retr = common.Retriever(binary, a.dir, a.prereg, pr["retrieval"]["k"], pr["retrieval"]["min_coverage"])
    examples, skipped, with_ctx = [], [], 0
    for r in train:
        ctx = retr(r["prompt"], exclude=[r["id"]])["context"]
        with_ctx += bool(ctx)
        ex = build_example(tok, r, lc["max_len"], ctx)
        (skipped.append((r["id"], ex)) if isinstance(ex, str) else examples.append(ex))
    for rid, why in skipped:
        print(f"skip {rid}: {why}")
    if not examples:
        die("no usable training examples")
    print(f"{with_ctx}/{len(train)} training prompts carry retrieval context")
    if len(skipped) > max(2, 0.05 * len(train)) and not a.smoke:
        die(f"{len(skipped)}/{len(train)} rows skipped (> 5%); fix max_len or the data before training")

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable") and a.precision != "fp32":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lcfg = peft.LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                           target_modules=lc["target_modules"], task_type="CAUSAL_LM")
    model = peft.get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lc["lr"], weight_decay=0.0)
    accum = lc["grad_accum"]
    total_steps = math.ceil(len(examples) * lc["epochs"] / accum)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    warm = max(1, total_steps // 10)

    def lr_at(s):  # linear warmup, then linear decay to zero
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (total_steps - s) / max(1, total_steps - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    dev = next(model.parameters()).device
    step, micro, t0, losses, curve = 0, 0, time.time(), [], []
    print(f"training {len(examples)} examples x {lc['epochs']} epochs, {total_steps} optimizer steps", flush=True)
    done = False
    for epoch in range(lc["epochs"]):
        order = list(range(len(examples)))
        random.shuffle(order)
        for i in order:
            ids, labels = examples[i]
            out = model(input_ids=torch.tensor([ids], device=dev), labels=torch.tensor([labels], device=dev))
            loss = out.loss / accum
            if not torch.isfinite(loss):
                die(f"non-finite loss at step {step}")
            loss.backward()
            losses.append(out.loss.item())
            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                mean = sum(losses[-accum:]) / len(losses[-accum:])
                curve.append(round(mean, 5))
                print(f"epoch {epoch + 1} step {step}/{total_steps} loss {mean:.4f} ({time.time() - t0:.0f}s)", flush=True)
                if step >= total_steps:
                    done = True
                    break
        if done:
            break
    if micro % accum and not done:  # flush the last partial accumulation
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    manifest = {
        "adapter_id": adapter_id,
        "base_model": pr.get("base_model"),
        "base_path": os.path.abspath(a.base),
        "base_hash": bh,
        "data_sha256": em["data_sha256"],
        "export_chain_head": em["chain_head"],
        "prereg_sha256": pr_sha,
        "thinking": "off",
        "train_precision": a.precision,
        "smoke": bool(a.smoke),
        "config": config,
        "n_examples": len(examples),
        "n_with_context": with_ctx,
        "skipped": [{"id": i, "why": w} for i, w in skipped],
        "lens_filter": ({"filter_sha256": lens_filter["filter_sha256"], "probe_sha256": lens_filter["probe_sha256"],
                         "n_dropped": len(lens_filter["dropped"]), "dropped": lens_filter["dropped"]}
                        if lens_filter else None),
        "loss_curve": curve,
        "seconds": round(time.time() - t0, 1),
        "versions": {"torch": torch.__version__, "peft": peft.__version__,
                     "transformers": __import__("transformers").__version__},
    }
    common.write_json(os.path.join(out_dir, "manifest.json"), manifest)
    print(f"\nadapter {adapter_id} -> {out_dir}")
    print(f"next: brain-adapt ledger register {out_dir}")


if __name__ == "__main__":
    main()
