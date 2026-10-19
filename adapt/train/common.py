"""Shared helpers for train_lora.py and eval_adapt.py.

Rules that must match the Rust side exactly live here and are tested against it:
normalize(), extract_decision() and the check semantics. Retrieval is never
reimplemented in Python; it calls `brain-adapt retrieve --json`, so training,
eval and the live loop all retrieve with the same code.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

DECISIONS = ("act", "answer", "ask", "abort")
PRECISIONS = ("bf16", "int8", "int8-t0", "fp32")  # fp32 is for CPU smoke tests only


def die(msg):
    sys.exit(f"error: {msg}")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip() and not l.lstrip().startswith("//")]


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ------------------------------------------------------------ rules shared with Rust


def normalize(s):
    """Same as store::normalize: lowercase, letters and digits kept, every other run -> one space."""
    out, space = [], True
    for c in s.lower():
        if c.isalnum():
            out.append(c)
            space = False
        elif not space:
            out.append(" ")
            space = True
    return "".join(out).rstrip()


_DEC = re.compile(r'"decision"\s*:\s*"(\w+)"')


def extract_decision(text):
    """Same as store::extract_decision: first "decision": "<word>" that is a real
    decision, else the whole trimmed text if it is a bare decision word."""
    for m in _DEC.finditer(text or ""):
        if m.group(1) in DECISIONS:
            return m.group(1)
    t = (text or "").strip().lower()
    return t if t in DECISIONS else None


def passes(check, output, correction):
    """True / False, or None when the row has no automatic check."""
    kind = check.get("type")
    if kind == "decision":
        return extract_decision(output) == check["value"]
    if kind == "contains":
        o = (output or "").lower()
        return all(s.lower() in o for s in check["value"])
    if kind == "exact":
        return normalize(output or "") == normalize(correction)
    return None


# ------------------------------------------------------------ prereg, binaries


def load_prereg(path):
    with open(path, encoding="utf-8") as f:
        pr = json.load(f)
    return pr, sha256_file(path)


def adapt_bin(arg):
    """The brain-adapt binary. Built by `cargo build --release` in adapt/."""
    if arg:
        return arg
    for c in ("adapt/target/release/brain-adapt.exe", "adapt/target/release/brain-adapt",
              "target/release/brain-adapt.exe", "target/release/brain-adapt"):
        if os.path.exists(c):
            return c
    die("brain-adapt binary not found; run `cargo build --release` in adapt/ or pass --adapt-bin")


def run_bin(binary, data_dir, prereg, *args):
    cmd = [binary, "--dir", data_dir, "--prereg", prereg, *args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        die(f"{' '.join(cmd)} failed:\n{r.stderr.strip()}")
    return r.stdout


def require_locked(binary, data_dir, prereg):
    out = run_bin(binary, data_dir, prereg, "prereg", "status").strip()
    if not out.startswith("locked and unchanged"):
        die(f"PREREG is not locked and unchanged ({out}). Run `brain-adapt prereg lock` first, "
            "or pass --smoke for a plumbing test that can never be promoted")


def require_export_current(data_dir, export_manifest):
    """Retrieval reads the live store. If corrections were added or deleted after
    the export, training/eval would retrieve from a different set than the one
    the dataset hash names. Refuse; re-export instead."""
    path = os.path.join(data_dir, "corrections.jsonl")
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    head = json.loads(last)["chain"] if last else None
    if head != export_manifest["chain_head"]:
        die("the correction store changed since this export (chain head differs); "
            "run `brain-adapt export` again and use the new folder")


class Retriever:
    """Calls the Rust retriever and caches by query."""

    def __init__(self, binary, data_dir, prereg, k, min_coverage):
        self.args = (binary, data_dir, prereg)
        self.k, self.min_coverage = k, min_coverage
        self.cache = {}

    def __call__(self, query, exclude=()):
        key = (query, tuple(exclude))
        if key not in self.cache:
            extra = [x for e in exclude for x in ("--exclude", e)]
            out = run_bin(*self.args, "retrieve", "--json", "--k", str(self.k),
                          "--min-coverage", str(self.min_coverage), *extra, "--", query)
            self.cache[key] = json.loads(out)
        return self.cache[key]


# ------------------------------------------------------------ model


def base_hash(model_path, cache_path=None):
    """Identity of the exact base weights (F7). Hashes config.json and every
    *.safetensors / *.bin weight file. Cached by (name, size, mtime) because a
    4B model is several GB."""
    files = sorted(f for f in os.listdir(model_path)
                   if f == "config.json" or f.endswith((".safetensors", ".bin")))
    if not any(f != "config.json" for f in files):
        die(f"no weight files in {model_path}")
    stat = [(f, os.path.getsize(os.path.join(model_path, f)), int(os.path.getmtime(os.path.join(model_path, f))))
            for f in files]
    key = sha256_text(canonical([os.path.abspath(model_path), stat]))
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        if key in cache:
            return cache[key]
    h = sha256_text(canonical([(f, sha256_file(os.path.join(model_path, f))) for f in files]))
    if cache_path:
        cache[key] = h
        write_json(cache_path, cache)
    return h


def load_model(model_path, precision, smoke=False):
    import torch
    import transformers

    if precision not in PRECISIONS:
        die(f"precision must be one of {PRECISIONS}")
    if precision == "fp32" and not smoke:
        die("fp32 is only for CPU smoke tests")
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    kw = {}
    if precision == "bf16":
        kw["dtype"] = torch.bfloat16
    elif precision in ("int8", "int8-t0"):
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=0.0 if precision == "int8-t0" else 6.0)
    else:
        kw["dtype"] = torch.float32
    if torch.cuda.is_available():
        kw["device_map"] = {"": 0}
    elif precision != "fp32":
        die(f"{precision} needs a CUDA GPU (torch.cuda.is_available() is False; wrong venv?)")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, **kw)
    model.eval()
    return tok, model


def messages_for(prompt, system=None, context=""):
    """The retrieval block goes in front of the system prompt. With no hits the
    messages are identical to the bare condition."""
    sys_text = "\n\n".join(x for x in (context.strip(), (system or "").strip()) if x)
    msgs = [{"role": "system", "content": sys_text}] if sys_text else []
    msgs.append({"role": "user", "content": prompt})
    return msgs


def render(tok, msgs, thinking):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=(thinking == "on"))
