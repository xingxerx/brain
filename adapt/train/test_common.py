"""Parity with the Rust rules. Run: python adapt/train/test_common.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import eval_adapt  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
cases = json.load(open(os.path.join(here, "..", "tests", "parity_cases.json"), encoding="utf-8"))
for raw, want in cases["normalize"]:
    assert common.normalize(raw) == want, (raw, common.normalize(raw), want)
for raw, want in cases["extract_decision"]:
    assert common.extract_decision(raw) == want, (raw, common.extract_decision(raw), want)

assert common.passes({"type": "decision", "value": "ask"}, '{"decision":"ask"}', "") is True
assert common.passes({"type": "contains", "value": ["42", "Total"]}, "the total is 42", "") is True
assert common.passes({"type": "contains", "value": ["43"]}, "the total is 42", "") is False
assert common.passes({"type": "exact"}, " The answer: 42. ", "the answer 42") is True
assert common.passes({"type": "none"}, "x", "y") is None

assert eval_adapt.mcnemar_one_sided(0, 0) == 1.0
assert abs(eval_adapt.mcnemar_one_sided(8, 0) - 1 / 256) < 1e-12
assert abs(eval_adapt.mcnemar_one_sided(5, 5) - 0.623046875) < 1e-9

m = common.messages_for("hi", None, "")
assert m == [{"role": "user", "content": "hi"}], m
m = common.messages_for("hi", "SYS", "CTX")
assert m[0] == {"role": "system", "content": "CTX\n\nSYS"}, m
h = [{"role": "assistant", "content": "A"}, {"role": "user", "content": "Observation: x"}]
m = common.messages_for("hi", "SYS", "", h)
assert [x["role"] for x in m] == ["system", "user", "assistant", "user"], m
print("python parity OK")

# ---- lens rules (no model needed)
import ast  # noqa: E402
import tempfile  # noqa: E402

import lens  # noqa: E402
import lens_filter  # noqa: E402

# The lens must never reach the loss. train_lora.py may read a filter FILE (a list of ids),
# but must not import lens code or read hidden states. If this fails, someone wired the
# probe into training: see PREREG_ADAPT.md, "The lens".
src = open(os.path.join(here, "train_lora.py"), encoding="utf-8").read()
imported = set()
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Import):
        imported |= {n.name.split(".")[0] for n in node.names}
    elif isinstance(node, ast.ImportFrom) and node.module:
        imported.add(node.module.split(".")[0])
assert not imported & {"lens", "lens_filter", "engine", "probe"}, f"train_lora.py imports lens code: {imported}"
assert "hidden_states" not in src, "train_lora.py reads hidden states"

P = {"act": 0.05, "answer": 0.92, "ask": 0.02, "abort": 0.01}
assert lens_filter.classify(P, "answer", 0.9) == "agree"
assert lens_filter.classify(P, "ask", 0.9) == "contradicts"
assert lens_filter.classify(P, "ask", 0.95) == "uncertain"
assert lens.top(P) == "answer"
assert lens.kl(P, P) == 0.0 and lens.kl(P, {"act": .25, "answer": .25, "ask": .25, "abort": .25}) > 0

body = {"export_chain_head": "h", "train_sha256": "t", "probe_sha256": "p", "prereg_sha256": "r",
        "settings": {"drop_confidence": 0.9}, "dropped": ["c_1"], "rows": []}
s = common.lens_filter_sha(body)
assert common.lens_filter_sha({**body, "rows": [{"x": 1}]}) == s, "per-row detail is not part of the identity"
assert common.lens_filter_sha({**body, "dropped": []}) != s, "editing the drop list must change the sha"


def fake_probe(thinking, precision="int8-t0", sys_text="SYS"):
    d = {"version": 1, "classes": list(lens.engine.DECISIONS), "layers": [0], "name": "L0", "lambda": 0.1,
         "mean": [0.0], "std": [1.0], "W": [[1.0], [0.0], [0.0], [-1.0]], "b": [0, 0, 0, 0],
         "cv": {"accuracy": 1.0},
         "trained_on": {"enable_thinking": thinking, "precision": precision,
                        "system_prompt_sha256": common.sha256_text(sys_text)}}
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(d, f)
    f.close()
    return f.name


for bad, why in ((fake_probe(True), "thinking-on probe"), (fake_probe(False, "bf16"), "precision mismatch")):
    try:
        lens.Lens(bad, "int8-t0")
        raise AssertionError(f"accepted a {why}")
    except SystemExit:
        pass
L = lens.Lens(fake_probe(False), "int8-t0")
dec = {"type": "decision", "value": "ask"}
assert L.na_reason({"check": dec, "system": "SYS"}) is None
assert L.na_reason({"check": dec, "system": "OTHER"}) == lens.NA_SYSTEM
assert L.na_reason({"check": dec, "system": "SYS", "history": [{"role": "user", "content": "x"}]}) == lens.NA_HISTORY
assert L.na_reason({"check": {"type": "contains", "value": ["x"]}, "system": "SYS"}) == lens.NA_NOT_DECISION
print("lens rules OK")
