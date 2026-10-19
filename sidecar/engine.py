"""Shared by server.py and probe.py: load Qwen3.5-4B, render prompts, read late-layer state.

Everything that decides *what the model sees* lives here, so the server and the probe can never
drift apart (same template, same thinking setting, same readout position).

Layer numbering: layer L means the output of decoder block L, 0-indexed. That is
`hidden_states[L + 1]` in transformers (index 0 is the embeddings). Qwen3.5-4B has 32 blocks,
so L runs 0..31. The last one (31) has the final norm applied; the others do not.

Readout position: the last prompt token, after the chat template's generation prompt, before
anything is generated. Same position as the v0 anchor gate (CONTRACT.md decision 1).
"""
import hashlib
import importlib.util
import json
import os
import sys

DECISIONS = ("act", "answer", "ask", "abort")
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_PATH = os.environ.get("QWEN_PATH", MODEL_ID)

# Precisions this code will load. Both int8 variants passed jspace-lab/readout/SPIKE_PREREG.md on
# the late band. int8 is bitsandbytes' default (fp16 outlier decomposition above 6.0); int8-t0
# turns that off: about 2x faster decode, slightly lower fidelity, passed narrowly. bf16 is the
# spike's own reference (needs about 10 GB of VRAM, so not the 8 GB laptop). nf4 was never tested.
INT8_THRESHOLDS = {"int8": 6.0, "int8-t0": 0.0}
VALIDATED = {"int8", "int8-t0", "bf16"}
KNOWN = VALIDATED | {"nf4"}
DEFAULT_PRECISION = os.environ.get("PRECISION", "int8-t0")


def fast_path_report():
    """Qwen3.5 is a hybrid model: 24 of its 32 blocks are Gated DeltaNet (linear attention).
    transformers runs those blocks with fused kernels only when these packages import;
    otherwise it falls back to a plain PyTorch loop, which is much slower per token."""
    return {
        "flash_linear_attention": importlib.util.find_spec("fla") is not None,
        "causal_conv1d": importlib.util.find_spec("causal_conv1d") is not None,
    }


def load_model(precision, allow_unvalidated=False):
    import warnings

    import torch
    import transformers

    # bitsandbytes repeats this on every forward pass (thousands of lines). It is expected: int8
    # matmuls run in fp16. Silence it so real output stays readable.
    warnings.filterwarnings("ignore", message=".*inputs will be cast from torch.bfloat16 to float16.*")
    import logging

    class _DropCast(logging.Filter):
        def filter(self, record):
            return "inputs will be cast from" not in record.getMessage()

    for name in ("bitsandbytes", "bitsandbytes.autograd._functions", ""):
        logging.getLogger(name).addFilter(_DropCast())
    for h in logging.getLogger().handlers:
        h.addFilter(_DropCast())

    if precision not in KNOWN:
        sys.exit(f"PRECISION={precision}: use one of {sorted(KNOWN)}")
    if precision not in VALIDATED and not allow_unvalidated:
        sys.exit(
            f"PRECISION={precision} did not pass the quantization spike. Pass "
            "--allow-unvalidated to load it anyway (dev speed tests only, never for PREREG runs)."
        )
    if not torch.cuda.is_available():
        sys.exit("CUDA not available; run jspace-lab/env_check.py")

    kw = {"dtype": torch.bfloat16, "device_map": {"": 0}, "attn_implementation": "sdpa"}
    if precision in INT8_THRESHOLDS:
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=INT8_THRESHOLDS[precision]
        )
    elif precision == "nf4":
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    print(f"loading {MODEL_PATH} as {precision} on GPU...", flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kw)
    hf.eval()
    print(f"weights on GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
    return tok, hf


def render(tok, messages, enable_thinking=None):
    """The model's own chat template. enable_thinking=None leaves the template default (on)."""
    kw = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kw["enable_thinking"] = bool(enable_thinking)
    return tok.apply_chat_template(messages, **kw)


def prompt_closes_thinking(text):
    """True when the rendered prompt already contains a closed (empty) think block, which is
    what Qwen templates emit for enable_thinking=False. Then the whole reply is the answer."""
    tail = text[text.rfind("<|im_start|>assistant"):]
    return "</think>" in tail


def last_token_states(hf, tok, text, layers):
    """{layer: float32 numpy vector} at the last prompt token, one forward pass, no generation."""
    import torch

    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(hf.device)
    with torch.inference_mode():
        out = hf(**ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    n_blocks = len(hs) - 1
    res = {}
    for L in layers:
        if not 0 <= L < n_blocks:
            raise ValueError(f"layer {L} out of range 0..{n_blocks - 1}")
        res[L] = hs[L + 1][0, -1].float().cpu().numpy()
    return res, int(ids["input_ids"].shape[1])


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- probe (numpy only)


class Probe:
    """Multinomial logistic regression on standardized late-layer state.

    features(): one layer, or the mean of a band of layers.
    """

    def __init__(self, d):
        import numpy as np

        self.d = d
        self.layers = list(d["layers"])
        self.mean = np.asarray(d["mean"], dtype=np.float32)
        self.std = np.asarray(d["std"], dtype=np.float32)
        self.W = np.asarray(d["W"], dtype=np.float32)  # [classes, hidden]
        self.b = np.asarray(d["b"], dtype=np.float32)
        self.classes = list(d["classes"])
        self.enable_thinking = d["trained_on"].get("enable_thinking")
        if self.classes != list(DECISIONS):
            raise ValueError(f"probe classes {self.classes} != {list(DECISIONS)}")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def features(self, states):
        import numpy as np

        return np.mean([states[L] for L in self.layers], axis=0)

    def predict(self, states):
        import numpy as np

        x = (self.features(states) - self.mean) / self.std
        z = self.W @ x + self.b
        z = z - z.max()
        p = np.exp(z) / np.exp(z).sum()
        top2 = np.sort(z)[-2:]
        return {c: float(v) for c, v in zip(self.classes, p)}, float(top2[1] - top2[0])

    def summary(self):
        t = self.d["trained_on"]
        return {
            "layers": self.layers,
            "lambda": self.d["lambda"],
            "cv_accuracy": self.d["cv"]["accuracy"],
            "trained_on": t.get("file"),
            "n": t.get("n"),
            "enable_thinking": t.get("enable_thinking"),
            "precision": t.get("precision"),
        }


def softmax_rows(Z):
    import numpy as np

    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_logreg(X, y, lam, iters=3000):
    """L2 multinomial logistic regression. X is standardized [n, h], y int labels [n].
    Returns W [k, h], b [k].

    Loss = mean cross-entropy + lam/2 * ||W||^2.

    With n (about 100) much smaller than h (2560), the optimal W lies in the row space of X, so
    we solve in that space (X = U S V^T, features Z = U S, at most n columns) and map back with
    W = Wz V^T. Exact, not an approximation. Nesterov gradient descent with step 1/L, where L
    bounds the curvature, so it converges and cannot diverge."""
    import numpy as np

    X = X.astype(np.float64)
    n, h = X.shape
    k = len(DECISIONS)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    keep = S > S.max() * 1e-6
    U, S, Vt = U[:, keep], S[keep], Vt[keep]
    Z = U * S  # [n, r]
    Y = np.eye(k)[y]
    L = 0.5 * (S.max() ** 2 / n + 1.0) + lam  # +1 covers the bias column
    step = 1.0 / L
    Wz = np.zeros((k, Z.shape[1]))
    b = np.zeros(k)
    Wz_prev, b_prev = Wz.copy(), b.copy()
    for t in range(1, iters + 1):
        mom = (t - 1) / (t + 2)
        Wy = Wz + mom * (Wz - Wz_prev)
        by = b + mom * (b - b_prev)
        G = (softmax_rows(Z @ Wy.T + by) - Y) / n
        Wz_prev, b_prev = Wz, b
        Wz = Wy - step * (G.T @ Z + lam * Wy)
        b = by - step * G.sum(axis=0)
    W = Wz @ Vt
    return W.astype(np.float32), b.astype(np.float32)


def logloss(P, y):
    import numpy as np

    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1.0))))
