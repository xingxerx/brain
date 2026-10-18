"""Lens sidecar v1: one process that both generates (Qwen3.5-4B, int8-t0) and gates from its internal state.

Serves, on one port (default 127.0.0.1:8765):
  GET  /health                 liveness, model, precision, gate mode, probe info, fast-path kernels
  GET  /v1/models              so `brain check` works with --base-url http://127.0.0.1:8765/v1
  POST /v1/chat/completions    OpenAI-style, non-streaming. Thinking on by default (the template's
                               default); `chat_template_kwargs.enable_thinking=false` turns it off.
                               Thinking goes to message.reasoning_content, the rest to content.
                               usage.timing has time-to-first-token and decode tokens/s.
  POST /gate                   {"messages": [...], "proposed": "act"} -> {"probs": {...}, "mode": ...}
                               Reads the model at the last prompt token, before any generation.

Gate modes:
  probe    (v1) a linear probe on late-layer state, trained by sidecar/probe.py on the probe split.
           One forward pass, no lens matrices on the GPU.
  anchors  (v0) J-lens logits in layers 20-30 scored against hand-picked anchor words. Kept for
           comparison. On dev it scored 6/24 against the text gate's 24/24 (2026-10-18 run 6).

Same weights, same chat template for generation and gating (CONTRACT.md decisions 2 and 3).

Run (PowerShell, jspace-lab venv, which has torch/transformers/bitsandbytes/jlens):
  $env:QWEN_PATH  = "C:\\models\\Qwen3.5-4B"
  $env:JLENS_PATH = "C:\\models\\jacobian-lens\\qwen3.5-4b\\jlens\\Salesforce-wikitext\\Qwen3.5-4B_jacobian_lens_n1000.pt"  # anchors mode only
  python sidecar\\server.py --probe sidecar\\probe.json      # probe gate
  python sidecar\\server.py                                  # v0 anchor gate
  python sidecar\\server.py --fake                           # no model, to test the wiring
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
from engine import DECISIONS, MODEL_ID  # noqa: E402

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
JLENS_PATH = os.environ.get("JLENS_PATH")
LATE_FIRST, LATE_LAST = (int(x) for x in os.environ.get("LATE_LAYERS", "20-30").split("-"))
# jlens truncates from the right (default 512), which would move "the last prompt token" into the
# middle of the prompt. Agent prompts reach ~800 tokens, so allow far more and refuse past it.
GATE_MAX_TOKENS = int(os.environ.get("GATE_MAX_TOKENS", "8192"))

# v0 anchor words, chosen before looking at any dev result. Common openers ("The", "It") are left
# out on purpose. Dev showed "answer" soaking up mass everywhere and "ask" firing on file tasks.
ANCHORS = {
    "act": ["act", "calculate", "compute", "read", "list", "tool", "run", "execute"],
    "answer": ["answer", "answered", "respond", "reply", "result"],
    "ask": ["ask", "clarify", "question", "which", "unclear", "specify"],
    "abort": ["abort", "refuse", "cannot", "decline", "stop", "unable", "unsafe"],
}
END_TOKENS = ("<|im_end|>", "<|endoftext|>")


class BadRequest(ValueError):
    """Turns into HTTP 400 with the message, so the Rust side can show why."""


def thinking_on(enable_thinking):
    return enable_thinking is None or bool(enable_thinking)


class FakeBackend:
    """No model. Lets the Rust loop and the HTTP layer be tested without a GPU."""

    precision = "fake"
    gate_mode = "fake"

    def health(self):
        return {"gate_mode": "fake"}

    def gate(self, messages, debug=False, enable_thinking=None):
        out = {
            "probs": {"act": 0.7, "answer": 0.1, "ask": 0.1, "abort": 0.1},
            "mode": "fake",
            "thinking": thinking_on(enable_thinking),
        }
        if debug:
            out["debug"] = {"note": "fake backend"}
        return out

    def chat(self, messages, max_tokens, temperature, enable_thinking=None):
        reasoning = "Thinking Process: this is a fake reply." if thinking_on(enable_thinking) else ""
        content = json.dumps(
            {
                "thought": "fake",
                "decision": "answer",
                "tool": "none",
                "input": "",
                "answer": "fake answer",
                "question": "",
            }
        )
        timing = {"ttft_ms": 1.0, "total_ms": 2.0, "decode_tok_s": 1000.0}
        return content, reasoning, "stop", (10, 12), timing


class QwenBackend:
    def __init__(self, precision, allow_unvalidated, gate_mode, probe_path, warmup=True):
        import torch
        import transformers

        self.torch = torch
        self.transformers = transformers
        fp = engine.fast_path_report()
        print(f"fast-path kernels for the DeltaNet layers: {fp}", flush=True)
        if not all(fp.values()):
            print(
                "  -> some kernels are missing; transformers falls back to a slow PyTorch loop for\n"
                "     24 of the 32 blocks. See README 'Speed' for the install commands.",
                flush=True,
            )
        self.fast_path = fp
        self.tok, self.hf = engine.load_model(precision, allow_unvalidated)
        self.precision = precision
        self.lock = threading.Lock()  # one GPU, one request at a time
        self.gate_mode = gate_mode
        self.probe = None
        self.lens = None
        if gate_mode == "probe":
            self.probe = engine.Probe.load(probe_path)
            trained = self.probe.d["trained_on"].get("precision")
            if trained != precision:
                sys.exit(f"probe was trained on {trained} activations, server runs {precision}")
            print(f"probe gate: {self.probe.summary()}", flush=True)
        else:
            import jlens

            self.jlens = jlens
            self.jmodel = jlens.from_hf(self.hf, self.tok)
            self.lens = self._load_lens(jlens)
            self.late = [l for l in self.lens.source_layers if LATE_FIRST <= int(l) <= LATE_LAST]
            if not self.late:
                sys.exit(f"lens has no layers in the late band {LATE_FIRST}-{LATE_LAST}")
            # Keep the late-band Jacobians resident on the GPU; jlens otherwise copies each one
            # host-to-device on every gate call.
            dev = self.jmodel.input_device
            for l in self.late:
                self.lens.jacobians[l] = self.lens.jacobians[l].to(dev)
            self.anchor_ids = self._anchor_ids()
            self.anchor_idx = {d: torch.tensor(v, device=dev) for d, v in self.anchor_ids.items()}
            print(f"anchor gate, token ids per class: { {d: len(v) for d, v in self.anchor_ids.items()} }")
        print(f"GPU memory in use: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
        if warmup:
            # First CUDA/bitsandbytes calls are slow; pay that at startup, not on the first task.
            t = time.perf_counter()
            msgs = [{"role": "user", "content": "Say hello."}]
            if self.probe is None or self.probe.enable_thinking is not False:
                self.gate(msgs)
            else:
                self.gate(msgs, enable_thinking=False)
            self.chat(msgs, 8, 0.0)
            print(f"warmed up in {time.perf_counter() - t:.1f}s", flush=True)

    def health(self):
        h = {"gate_mode": self.gate_mode, "fast_path": self.fast_path}
        if self.probe is not None:
            h["probe"] = self.probe.summary()
        return h

    # ------------------------------------------------------------ anchors (v0)

    def _load_lens(self, jlens):
        if not JLENS_PATH:
            return jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
        for name in ("load", "from_file", "from_path"):
            fn = getattr(jlens.JacobianLens, name, None)
            if callable(fn):
                return fn(JLENS_PATH)
        import huggingface_hub

        def local(*a, **k):
            return JLENS_PATH

        huggingface_hub.hf_hub_download = local
        for m in list(sys.modules.values()):
            if m and getattr(m, "__name__", "").startswith("jlens") and hasattr(m, "hf_hub_download"):
                m.hf_hub_download = local
        return jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)

    def _anchor_ids(self):
        out = {}
        for d, words in ANCHORS.items():
            ids = set()
            for w in words:
                for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                    enc = self.tok.encode(v, add_special_tokens=False)
                    if len(enc) == 1:
                        ids.add(enc[0])
            out[d] = sorted(ids)
        return out

    def _gate_anchors(self, text, debug):
        # Same readout as JacobianLens.apply(positions=[-1]) on the late band, but it stays on the
        # GPU end to end (apply() moves every layer's full-vocab logits to the CPU) and skips the
        # final-layer logits we never use.
        torch = self.torch
        with self.lock, torch.no_grad():
            ids = self.jmodel.encode(text, max_length=GATE_MAX_TOKENS + 1)
            if ids.shape[1] > GATE_MAX_TOKENS:
                raise BadRequest(f"prompt longer than GATE_MAX_TOKENS={GATE_MAX_TOKENS}")
            with self.jlens.ActivationRecorder(self.jmodel.layers, at=self.late) as rec:
                self.jmodel.forward(ids)
                acts = {l: rec.activations[l][0, -1:].detach().float() for l in self.late}
            probs_by_layer = {}
            for l in self.late:
                logits = self.jmodel.unembed(self.lens.transport(acts[l], l)).float()
                probs_by_layer[int(l)] = torch.softmax(logits[0], dim=-1)
            mass_t = {
                d: torch.stack([p[self.anchor_idx[d]].sum() for p in probs_by_layer.values()]).mean()
                for d in DECISIONS
            }
            mass = {d: float(m) for d, m in mass_t.items()}  # one sync per decision, at the end
        used = len(probs_by_layer)
        last = (max(probs_by_layer), probs_by_layer[max(probs_by_layer)])
        total = sum(mass.values())
        probs = {d: (m / total if total > 0 else 0.25) for d, m in mass.items()}
        out = {"probs": probs, "mode": "anchors", "anchor_mass": total}
        if debug:
            layer, p = last
            top = p.topk(10)
            out["debug"] = {
                "layer": layer,
                "layers_used": used,
                "raw_mass": mass,
                "top10": [[self.tok.decode([int(i)]), round(float(v), 4)] for v, i in zip(top.values, top.indices)],
            }
        return out

    # ------------------------------------------------------------ gate

    def gate(self, messages, debug=False, enable_thinking=None):
        t0 = time.perf_counter()
        text = engine.render(self.tok, messages, enable_thinking)
        if self.gate_mode == "probe":
            want = self.probe.enable_thinking
            if want is not None and want != thinking_on(enable_thinking):
                raise BadRequest(
                    f"probe was trained with thinking {'on' if want else 'off'}, this request has it "
                    f"{'on' if thinking_on(enable_thinking) else 'off'}. Retrain the probe or match --no-think."
                )
            with self.lock:
                states, n_tok = engine.last_token_states(self.hf, self.tok, text, self.probe.layers)
            probs, margin = self.probe.predict(states)
            out = {"probs": probs, "mode": "probe", "layers": self.probe.layers, "margin": margin}
            if debug:
                out["debug"] = {"prompt_tokens": n_tok}
        else:
            out = self._gate_anchors(text, debug)
        out["thinking"] = thinking_on(enable_thinking)
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    # ------------------------------------------------------------ chat

    def chat(self, messages, max_tokens, temperature, enable_thinking=None):
        torch = self.torch

        class Clock(self.transformers.StoppingCriteria):
            """Never stops anything; records when the first new token exists."""

            first = None

            def __call__(self, input_ids, scores, **kw):
                if self.first is None:
                    self.first = time.perf_counter()
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        text = engine.render(self.tok, messages, enable_thinking)
        inputs = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.hf.device)
        clock = Clock()
        kw = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "stopping_criteria": self.transformers.StoppingCriteriaList([clock]),
        }
        if temperature > 0:
            kw["temperature"] = temperature
        with self.lock, torch.inference_mode():
            t0 = time.perf_counter()
            out = self.hf.generate(**inputs, **kw)
            t1 = time.perf_counter()
        new = out[0][inputs["input_ids"].shape[1]:]
        n_in, n_out = int(inputs["input_ids"].shape[1]), int(new.shape[0])
        first = clock.first or t1
        decode_s = max(t1 - first, 1e-6)
        timing = {
            "ttft_ms": round((first - t0) * 1000, 1),
            "total_ms": round((t1 - t0) * 1000, 1),
            "decode_tok_s": round(max(n_out - 1, 0) / decode_s, 2),
        }
        print(
            f"chat: {n_in} in, {n_out} out, {timing['total_ms'] / 1000:.1f}s, "
            f"{timing['decode_tok_s']} tok/s decode, first token {timing['ttft_ms']:.0f} ms",
            flush=True,
        )
        raw = self.tok.decode(new, skip_special_tokens=False)
        for t in END_TOKENS:
            raw = raw.replace(t, "")
        raw = raw.strip()
        if engine.prompt_closes_thinking(text):
            # Thinking off: the template already closed an empty think block. All of it is the answer.
            return raw, "", "stop", (n_in, n_out), timing
        if "</think>" in raw:
            reasoning, _, content = raw.rpartition("</think>")
            return content.strip(), reasoning.replace("<think>", "").strip(), "stop", (n_in, n_out), timing
        # Never closed the thinking block (hit max_tokens): no usable answer, say so honestly.
        return "", raw.replace("<think>", "").strip(), "length", (n_in, n_out), timing


def make_handler(backend):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "model": MODEL_ID, "precision": backend.precision, **backend.health()})
            elif self.path in ("/v1/models", "/models"):
                self._send(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                b = self._body()
                msgs = b.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    return self._send(400, {"error": "messages must be a non-empty list"})
                think = (b.get("chat_template_kwargs") or {}).get("enable_thinking")
                if self.path == "/gate":
                    return self._send(200, backend.gate(msgs, bool(b.get("debug")), think))
                if self.path in ("/v1/chat/completions", "/chat/completions"):
                    content, reasoning, finish, (n_in, n_out), timing = backend.chat(
                        msgs, int(b.get("max_tokens") or 2048), float(b.get("temperature") or 0.0), think
                    )
                    return self._send(
                        200,
                        {
                            "id": f"chatcmpl-{int(time.time() * 1000)}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": MODEL_ID,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": content,
                                        "reasoning_content": reasoning,
                                    },
                                    "finish_reason": finish,
                                }
                            ],
                            "usage": {
                                "prompt_tokens": n_in,
                                "completion_tokens": n_out,
                                "total_tokens": n_in + n_out,
                                "timing": timing,
                            },
                        },
                    )
                self._send(404, {"error": "not found"})
            except BadRequest as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # keep the server up; report the error to the caller
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}", flush=True)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="no model; canned replies")
    ap.add_argument("--host", default=os.environ.get("SIDECAR_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SIDECAR_PORT", "8765")))
    ap.add_argument("--precision", default=engine.DEFAULT_PRECISION,
                    help="int8-t0 (default, ~2x faster decode) or int8: both passed the spike and fit 8 GB. "
                         "bf16: the spike's reference, ~10 GB. nf4: never tested")
    ap.add_argument("--allow-unvalidated", action="store_true",
                    help="allow precisions the spike did not validate (speed tests only)")
    ap.add_argument("--probe", default=os.environ.get("PROBE_PATH"),
                    help="probe.json from sidecar/probe.py train; turns on the probe gate")
    ap.add_argument("--gate-mode", choices=["probe", "anchors"],
                    help="default: probe when --probe is given, else anchors")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()
    mode = a.gate_mode or ("probe" if a.probe else "anchors")
    if mode == "probe" and not a.probe:
        sys.exit("--gate-mode probe needs --probe path\\to\\probe.json")
    if a.fake:
        backend = FakeBackend()
    else:
        backend = QwenBackend(a.precision, a.allow_unvalidated, mode, a.probe, warmup=not a.no_warmup)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(backend))
    print(f"lens sidecar ({backend.precision}, gate {backend.gate_mode}) listening on http://{a.host}:{a.port}",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
