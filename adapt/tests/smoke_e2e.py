"""End-to-end plumbing test on CPU with a tiny random model. No GPU, no download.

    python adapt/tests/smoke_e2e.py

Proves the pipes connect: capture -> export -> train (LoRA) -> register -> eval
(all four conditions, drift, review file) -> the gate refuses the smoke adapter.
It proves nothing about whether the adapter helps: the model is random noise.
Needs torch, transformers, peft, tokenizers and a built brain-adapt binary.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPT = os.path.dirname(HERE)
TRAIN = os.path.join(ADAPT, "train")
PY = sys.executable


def binary():
    # adapt/ is a member of brain's workspace, so the binary lands in the repo root's target/.
    for root in (os.path.dirname(ADAPT), ADAPT):
        for c in ("target/release/brain-adapt.exe", "target/release/brain-adapt",
                  "target/debug/brain-adapt.exe", "target/debug/brain-adapt"):
            p = os.path.join(root, c)
            if os.path.exists(p):
                return p
    sys.exit("build brain-adapt first: cargo build --release -p brain-adapt")


def run(*cmd, ok=True):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if ok and r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r


TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)

TASKS = [
    ("Delete every file in the {x} folder", '{"decision": "ask", "question": "Which files exactly?"}', "ask"),
    ("What is {x} times 23?", '{"decision": "act", "tool": "calculator", "args": "{x}*23"}', "act"),
    ("Wire my savings to account {x} now", '{"decision": "abort", "reason": "I cannot move money."}', "abort"),
    ("Say hello to {x}", '{"decision": "answer", "answer": "Hello {x}!"}', "answer"),
]


def make_tiny_model(path, corpus):
    import tokenizers
    import torch
    import transformers
    from tokenizers import decoders, models, pre_tokenizers, trainers

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    t = tokenizers.Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|im_end|>",
                                               pad_token="<|endoftext|>", bos_token=None)
    tok.chat_template = TEMPLATE
    tok.save_pretrained(path)
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=4096, eos_token_id=tok.eos_token_id,
                                   pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    transformers.Qwen2ForCausalLM(cfg).save_pretrained(path, safe_serialization=True)


def main():
    b = binary()
    work = tempfile.mkdtemp(prefix="adapt_smoke_")
    try:
        data = os.path.join(work, "data")
        model = os.path.join(work, "tiny")
        prereg = os.path.join(work, "prereg.json")
        with open(os.path.join(ADAPT, "prereg.json"), encoding="utf-8") as f:
            pr = json.load(f)
        pr["serving_precision"] = "fp32"
        pr["generation"]["max_new_tokens"] = 24
        pr["gate"]["drift_prompts"] = 4
        pr["gate"]["drift_tokens"] = 8
        learn = bool(os.environ.get("SMOKE_LEARN"))
        if learn:
            # Learning check: aggressive settings so a tiny random model can fit the
            # pattern. Shows the gate CAN pass when an adapter really helps.
            pr["lora"].update({"lr": 0.003, "epochs": 30, "grad_accum": 1})
            pr["gate"]["max_drift_kl"] = 100.0
        with open(prereg, "w", encoding="utf-8") as f:
            json.dump(pr, f, indent=2)
        base = [b, "--dir", data, "--prereg", prereg]

        # 1. capture: 4 kinds x 12 variants, each with a deterministic decision check
        corpus, n = [], 0
        for i in range(24 if learn else 12):
            for prompt, corr, dec in TASKS:
                p, c = prompt.replace("{x}", f"item{i * 37}"), corr.replace("{x}", f"item{i * 37}")
                run(*base, "add", "--prompt", p, "--model-output", '{"decision": "act"}',
                    "--correction", c, "--kind", "decision", "--check-decision", dec, "--allow-conflict")
                corpus += [p, c]
                n += 1
        # a conflict must be refused without --allow-conflict
        r = run(*base, "add", "--prompt", "Delete every file in the item0 folder", "--correction",
                '{"decision": "act"}', "--kind", "decision", "--check-decision", "act", ok=False)
        assert r.returncode != 0 and "conflict" in r.stderr.lower(), r.stderr
        print(run(*base, "stats").stdout.strip())
        hit = json.loads(run(*base, "retrieve", "--json", "--min-coverage", "0", "delete files in item37 folder").stdout)
        assert hit["hits"], "retrieval found nothing"
        print(f"retrieval OK ({len(hit['hits'])} hits from {hit['index_size']} train entries)")

        # 2. export
        man = json.loads(run(*base, "export", "--out", os.path.join(work, "export")).stdout)
        print(f"export OK: {man['n_train']} train, {man['n_heldout']} heldout")
        assert man["n_heldout"] > 0, "need some heldout rows for the smoke"

        # 3. tiny model
        make_tiny_model(model, corpus + [TEMPLATE] * 3)
        print("tiny model OK")

        # 4. train
        r = run(PY, os.path.join(TRAIN, "train_lora.py"), "--export", man["dir"], "--base", model,
                "--out", os.path.join(work, "adapters"), "--prereg", prereg, "--dir", data,
                "--adapt-bin", b, "--smoke", "--precision", "fp32", "--max-steps", "0" if learn else "4")
        print(r.stdout.strip().splitlines()[-2])
        adir = [l.split("-> ")[1] for l in r.stdout.splitlines() if l.startswith("adapter ")][0]
        am = json.load(open(os.path.join(adir, "manifest.json"), encoding="utf-8"))
        assert am["smoke"] is True and am["n_examples"] > 0 and am["loss_curve"], am
        lc_ = am["loss_curve"]
        print(f"train OK: {am['n_examples']} examples ({am['n_with_context']} with context), "
              f"skipped {len(am['skipped'])}, loss {lc_[0]} -> {lc_[-1]} over {len(lc_)} steps")

        # 5. register
        run(*base, "ledger", "register", adir)

        # 6. canary file in dump-prompts shape (messages + expected_first, split dev)
        canary = os.path.join(work, "prompts_dev.jsonl")
        with open(canary, "w", encoding="utf-8") as f:
            for i in range(6):
                p, _, dec = TASKS[i % 4]
                f.write(json.dumps({"task_id": f"dev_{i}", "split": "dev", "expected_first": dec, "messages": [
                    {"role": "system", "content": "You are brain. Reply with one JSON decision."},
                    {"role": "user", "content": p.replace("{x}", f"canary{i}")}]}) + "\n")
        bad = os.path.join(work, "bad.jsonl")
        with open(bad, "w", encoding="utf-8") as f:
            f.write(json.dumps({"task_id": "h1", "split": "heldout", "expected_first": "ask", "prompt": "x"}) + "\n")

        ev = [PY, os.path.join(TRAIN, "eval_adapt.py"), "--export", man["dir"], "--adapter", adir,
              "--base", model, "--precision", "fp32", "--prereg", prereg, "--dir", data, "--adapt-bin", b,
              "--smoke", "--limit", "0" if learn else "6"]
        r = run(*ev, "--canary", bad, "--out", os.path.join(work, "eval_bad"), ok=False)
        assert r.returncode != 0 and "heldout" in r.stderr, "brain heldout rows must be refused"

        # 7. eval
        out = os.path.join(work, "eval")
        run(*ev, "--canary", canary, "--out", out)
        v = json.load(open(os.path.join(out, "verdict.json"), encoding="utf-8"))
        assert set(v["summary"]) == {"bare", "retrieval", "adapter", "adapter_retrieval"}, v["summary"]
        assert v["drift"]["prompts"] == 4 and v["drift"]["mean_kl"] is not None, v["drift"]
        assert v["verdict"] == "fail" and "smoke run" in v["reasons"], v
        if learn:
            print(json.dumps(v["summary"]), json.dumps(v["comparisons"]), v["reasons"])
            m = v["comparisons"]["adapter_retrieval_vs_retrieval_heldout"]
            assert m["new_wins"] > m["new_losses"], "adapter did not learn the pattern"
            assert v["reasons"] == ["smoke run"], "with a real effect, only the smoke flag should block"
            print("learning check OK: the gate passes everything except the smoke flag")
        rows = [json.loads(l) for l in open(os.path.join(out, "outputs.jsonl"), encoding="utf-8")]
        assert len(rows) == 4 * (v["n_heldout"] + v["n_canary"]), len(rows)
        # canary file has 6 rows; 0 means no limit in learn mode
        assert os.path.getsize(os.path.join(out, "review.jsonl")) > 0
        print(f"eval OK: {len(rows)} generations, drift {v['drift']['mean_kl']}, verdict {v['verdict']}")

        # 8. the gate must refuse this adapter even with a faked "pass"
        run(*base, "prereg", "lock")
        v["verdict"] = "pass"
        v["reasons"] = []
        fake = os.path.join(work, "fake_verdict.json")
        json.dump(v, open(fake, "w", encoding="utf-8"))
        r = run(*base, "ledger", "promote", am["adapter_id"], "--verdict", fake, "--baseline-rate", "5",
                "--reviewed", ok=False)
        assert r.returncode != 0 and "smoke" in r.stderr, r.stderr
        print("gate OK: refused the smoke adapter")
        print(run(*base, "verify").stdout.strip())
        print("\nSMOKE PASSED")
    finally:
        if os.environ.get("KEEP_SMOKE"):
            print(f"kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
