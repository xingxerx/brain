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
