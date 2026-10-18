"""Linear probe for the brain gate: does late-layer state at the last prompt token encode the
right first move (act / answer / ask / abort)?

Three commands (PowerShell, from the brain folder, jspace-lab venv):

  # 1. Export the exact step-1 prompts the Rust loop would send (same system prompt).
  cargo run --release -- dump-prompts --split probe --out runs\\prompts_probe.jsonl
  cargo run --release -- dump-prompts --tasks eval\\tasks.jsonl --split dev --out runs\\prompts_dev.jsonl

  # 2. One forward pass per prompt, no generation. Saves every layer's last-token state.
  python sidecar\\probe.py collect --prompts runs\\prompts_probe.jsonl --out runs\\acts_probe.npz
  python sidecar\\probe.py collect --prompts runs\\prompts_dev.jsonl   --out runs\\acts_dev.npz

  # 3. Train on the probe split ONLY (5-fold CV picks layer and regularization), then score dev.
  python sidecar\\probe.py train --acts runs\\acts_probe.npz --out sidecar\\probe.json
  python sidecar\\probe.py score --probe sidecar\\probe.json --acts runs\\acts_dev.npz

Rules this script enforces so the result means something:
  - train refuses any file that is not split=probe. Dev and heldout never touch training or
    selection. Heldout is never collected by these instructions at all.
  - Every candidate (layer, lambda) is scored by the same stratified 5-fold CV with a fixed seed.
  - Two baselines run on the same folds: majority class, and a bag-of-words classifier on the
    user's request text. If the probe doesn't beat bag-of-words, it is reading the words in the
    prompt, not anything deeper.
  - A label-shuffled run of the chosen config shows what chance looks like with this many
    features and this few examples.
  - collect and the server render prompts with the same engine.render(), and the probe file
    records the thinking setting; the server refuses to gate with a mismatched setting.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
from engine import DECISIONS  # noqa: E402

SEED = 1234
FOLDS = 5
LAMBDAS = (0.01, 0.1, 1.0, 10.0)
DEFAULT_LAYERS = "12-31"
BAND = (20, 30)  # the v0 late band, scored as one extra candidate (mean of these layers)


def parse_layers(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip() and not l.lstrip().startswith("//")]


# ------------------------------------------------------------------ collect


def cmd_collect(a):
    import numpy as np

    rows = read_jsonl(a.prompts)
    if not rows:
        sys.exit(f"no prompts in {a.prompts}")
    splits = sorted({r["split"] for r in rows})
    if "heldout" in splits and not a.i_locked_the_prereg:
        sys.exit("refusing to collect heldout prompts before PREREG is locked "
                 "(pass --i-locked-the-prereg if it is)")
    tok, hf = engine.load_model(a.precision, a.allow_unvalidated)
    cfg = hf.config.get_text_config() if hasattr(hf.config, "get_text_config") else hf.config
    n_blocks = cfg.num_hidden_layers
    layers = list(range(n_blocks))
    think = False if a.no_think else None
    states = np.zeros((len(rows), n_blocks, cfg.hidden_size), np.float16)
    t0 = time.time()
    for i, r in enumerate(rows):
        text = engine.render(tok, r["messages"], think)
        st, n_tok = engine.last_token_states(hf, tok, text, layers)
        for L in layers:
            states[i, L] = st[L]
        if i == 0:
            print(f"prompt tokens: {n_tok}; readout token: {tok.decode(tok(text)['input_ids'][-1:])!r}")
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"  {i + 1}/{len(rows)}  {time.time() - t0:.1f}s", flush=True)
    meta = {
        "file": os.path.basename(a.prompts),
        "splits": splits,
        "n": len(rows),
        "model": engine.MODEL_ID,
        "precision": a.precision,
        "enable_thinking": not a.no_think,
        "system_prompt_sha256": engine.sha256_text(rows[0]["messages"][0]["content"]),
        "n_blocks": n_blocks,
    }
    np.savez_compressed(
        a.out,
        states=states,
        ids=np.array([r["id"] for r in rows]),
        labels=np.array([DECISIONS.index(r["expected_first"]) for r in rows]),
        prompts=np.array([r["messages"][-1]["content"] for r in rows]),
        meta=json.dumps(meta),
    )
    print(f"saved {a.out}: {len(rows)} prompts x {n_blocks} layers ({time.time() - t0:.1f}s)")


def load_acts(path):
    import numpy as np

    z = np.load(path, allow_pickle=False)
    return {
        "states": z["states"].astype(np.float32),
        "ids": [str(x) for x in z["ids"]],
        "y": z["labels"].astype(int),
        "prompts": [str(x) for x in z["prompts"]],
        "meta": json.loads(str(z["meta"])),
    }


# ------------------------------------------------------------------ train


def stratified_folds(y, k, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def features(states, layers):
    return states[:, layers, :].mean(axis=1)


def standardize(Xtr, Xte):
    import numpy as np

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    # A feature that is constant in training (a word never seen) must not blow up at test time.
    sd = np.where(sd < 1e-3, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd


def cv_score(X, y, fold, lam):
    import numpy as np

    P = np.zeros((len(y), len(DECISIONS)))
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        Xtr, Xte, _, _ = standardize(X[tr], X[te])
        W, b = engine.fit_logreg(Xtr, y[tr], lam)
        P[te] = engine.softmax_rows(Xte @ W.T + b)
    return float((P.argmax(1) == y).mean()), engine.logloss(P, y), P


WORD = re.compile(r"[a-z0-9_.]+")


def bow(prompts, vocab=None):
    import numpy as np

    toks = [set(WORD.findall(p.lower())) for p in prompts]
    if vocab is None:
        vocab = sorted({w for t in toks for w in t})
    index = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(prompts), max(1, len(vocab))), np.float32)
    for i, t in enumerate(toks):
        for w in t:
            if w in index:
                X[i, index[w]] = 1.0
    return X, vocab


def cmd_train(a):
    import numpy as np

    d = load_acts(a.acts)
    if d["meta"]["splits"] != ["probe"]:
        sys.exit(f"train only accepts split=probe activations; {a.acts} has {d['meta']['splits']}")
    S, y = d["states"], d["y"]
    counts = np.bincount(y, minlength=len(DECISIONS))
    print(f"{len(y)} probe tasks: " + ", ".join(f"{c}={n}" for c, n in zip(DECISIONS, counts)))
    fold = stratified_folds(y, FOLDS, SEED)

    layers = [L for L in parse_layers(a.layers) if L < S.shape[1]]
    candidates = [((L,), f"L{L}") for L in layers]
    band = tuple(L for L in range(BAND[0], BAND[1] + 1) if L < S.shape[1])
    candidates.append((band, f"band{BAND[0]}-{BAND[1]}"))

    results = []
    t0 = time.time()
    for lay, name in candidates:
        X = features(S, list(lay))
        for lam in LAMBDAS:
            acc, nll, _ = cv_score(X, y, fold, lam)
            results.append({"name": name, "layers": list(lay), "lambda": lam, "acc": acc, "nll": nll})
        best_here = max((r for r in results if r["name"] == name), key=lambda r: (r["acc"], -r["nll"]))
        print(f"  {name:<10} best lambda {best_here['lambda']:<5} CV acc {best_here['acc']:.3f}  "
              f"nll {best_here['nll']:.3f}", flush=True)
    best = max(results, key=lambda r: (r["acc"], -r["nll"]))

    # Baselines on the same folds.
    majority = float(counts.max() / counts.sum())
    Xb, _ = bow(d["prompts"])
    bow_acc = max(cv_score(Xb, y, fold, lam)[0] for lam in LAMBDAS)
    rng = np.random.default_rng(SEED + 1)
    y_shuf = rng.permutation(y)
    shuf_acc, _, _ = cv_score(features(S, best["layers"]), y_shuf, fold, best["lambda"])

    # Final fit on all probe tasks with the chosen config.
    X = features(S, best["layers"])
    Xs, _, mu, sd = standardize(X, X[:1])
    W, b = engine.fit_logreg(Xs, y, best["lambda"])
    _, _, P = cv_score(X, y, fold, best["lambda"])
    conf = np.zeros((len(DECISIONS), len(DECISIONS)), int)
    for t, p in zip(y, P.argmax(1)):
        conf[t, p] += 1

    out = {
        "version": 1,
        "classes": list(DECISIONS),
        "layers": best["layers"],
        "name": best["name"],
        "lambda": best["lambda"],
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "W": W.tolist(),
        "b": b.tolist(),
        "cv": {
            "folds": FOLDS,
            "seed": SEED,
            "accuracy": best["acc"],
            "nll": best["nll"],
            "confusion": {DECISIONS[i]: dict(zip(DECISIONS, map(int, conf[i]))) for i in range(4)},
            "baselines": {
                "majority": majority,
                "bag_of_words": bow_acc,
                "shuffled_labels": shuf_acc,
            },
            "all_candidates": results,
        },
        "trained_on": {**d["meta"], "file": os.path.basename(a.acts)},
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"\nchosen: {best['name']} lambda={best['lambda']}  CV accuracy {best['acc']:.3f} "
          f"({time.time() - t0:.0f}s)")
    print(f"baselines on the same folds: majority {majority:.3f} | bag-of-words {bow_acc:.3f} | "
          f"shuffled labels {shuf_acc:.3f}")
    print("CV confusion (rows = expected, cols = probe):")
    print("          " + " ".join(f"{c:>7}" for c in DECISIONS))
    for i, c in enumerate(DECISIONS):
        print(f"  {c:<7} " + " ".join(f"{v:>7}" for v in conf[i]))
    if best["acc"] <= bow_acc:
        print("\nWARNING: the probe does not beat bag-of-words. It may only be reading the words "
              "in the request. Say so in any write-up.")
    print(f"\nwrote {a.out}")


# ------------------------------------------------------------------ score


def cmd_score(a):
    import numpy as np

    probe = engine.Probe.load(a.probe)
    d = load_acts(a.acts)
    if "probe" in d["meta"]["splits"]:
        print("note: scoring probe-split activations, i.e. training data. This is not a test.")
    for key in ("precision", "enable_thinking", "system_prompt_sha256"):
        want, got = probe.d["trained_on"].get(key), d["meta"].get(key)
        if want != got:
            sys.exit(f"mismatch on {key}: probe trained with {want!r}, activations have {got!r}")
    S, y = d["states"], d["y"]
    preds, margins = [], []
    for i in range(len(y)):
        st = {L: S[i, L] for L in probe.layers}
        p, m = probe.predict(st)
        preds.append(DECISIONS.index(max(p, key=p.get)))
        margins.append(m)
    preds = np.array(preds)
    acc = float((preds == y).mean())

    # Bag-of-words baseline trained on the probe split, scored on the same tasks.
    probe_acts = a.bow_train
    bow_line = ""
    if probe_acts and os.path.exists(probe_acts):
        tr = load_acts(probe_acts)
        Xtr, vocab = bow(tr["prompts"])
        Xte, _ = bow(d["prompts"], vocab)
        best = None
        for lam in LAMBDAS:
            Xtr_s, Xte_s, _, _ = standardize(Xtr, Xte)
            W, b = engine.fit_logreg(Xtr_s, tr["y"], lam)
            acc_b = float((engine.softmax_rows(Xte_s @ W.T + b).argmax(1) == y).mean())
            best = acc_b if best is None else max(best, acc_b)
        bow_line = f" | bag-of-words (trained on probe split): {best:.3f}"

    print(f"{probe.d['name']} on {os.path.basename(a.acts)} ({', '.join(d['meta']['splits'])}): "
          f"accuracy {acc:.3f} ({int((preds == y).sum())}/{len(y)}){bow_line}")
    for c in range(4):
        mask = y == c
        if mask.any():
            print(f"  {DECISIONS[c]:<7} {int((preds[mask] == c).sum())}/{int(mask.sum())}")
    print("\nper task:")
    for i in range(len(y)):
        mark = "ok " if preds[i] == y[i] else "XX "
        print(f"  {mark}{d['ids'][i]:<16} expected {DECISIONS[y[i]]:<7} probe {DECISIONS[preds[i]]:<7} "
              f"margin {margins[i]:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="one forward pass per prompt, save every layer's last-token state")
    c.add_argument("--prompts", required=True, help="JSONL from `brain dump-prompts`")
    c.add_argument("--out", required=True)
    c.add_argument("--precision", default=engine.DEFAULT_PRECISION,
                   help="must match the server's precision (default int8-t0, same as the server)")
    c.add_argument("--allow-unvalidated", action="store_true")
    c.add_argument("--no-think", action="store_true", help="render with enable_thinking=False")
    c.add_argument("--i-locked-the-prereg", action="store_true")

    t = sub.add_parser("train", help="CV over layers and lambdas on the probe split, write probe.json")
    t.add_argument("--acts", required=True)
    t.add_argument("--out", default=os.path.join("sidecar", "probe.json"))
    t.add_argument("--layers", default=DEFAULT_LAYERS, help="candidate layers, e.g. 12-31 or 20,24,28")

    s = sub.add_parser("score", help="apply a trained probe to collected activations (e.g. dev)")
    s.add_argument("--probe", required=True)
    s.add_argument("--acts", required=True)
    s.add_argument("--bow-train", default=os.path.join("runs", "acts_probe.npz"),
                   help="probe-split activations, used to train the bag-of-words baseline")

    a = ap.parse_args()
    {"collect": cmd_collect, "train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
