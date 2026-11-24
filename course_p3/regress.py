"""Regression check for Project 3 (stand-in for the course's course/regress.py).

    python course_p3/regress.py record   # before editing: save baseline outputs
    python course_p3/regress.py check    # after editing: every number must match exactly

Covers the four course models (Gemma without its window block, which the course branch
refuses) plus the Project 3 evaluation grid on Llama-3.1-8B with the feature off.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runner import REPO, make_model_config, run_config  # noqa: E402

MC = os.path.join(REPO, "configs/model-config/course_project")
BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.json")

CASES = {
    "llama8b_default": dict(base=f"{MC}/Llama3.1-8B_inf.yaml", batch=1, prompt=1024, output=1024),
    "qwen3_30b_default": dict(base=f"{MC}/qwen3_30b_a3b_inf.yaml", batch=None),
    "mimo_7b_default": dict(base=f"{MC}/mimo_7b_inf.yaml", batch=None),
    "gemma3_4b_dense": dict(base=f"{MC}/gemma_3_4b_inf.yaml", batch=None, drop_window=True),
}
for b in (1, 16, 64):
    for o in (128, 256):
        CASES[f"llama8b_b{b}_s512_o{o}"] = dict(base=f"{MC}/Llama3.1-8B_inf.yaml", batch=b, prompt=512, output=o)
CASES["llama8b_b8_s2048_o128_sample16"] = dict(
    base=f"{MC}/Llama3.1-8B_inf.yaml", batch=8, prompt=2048, output=128, sample_every=16
)

_DROP = re.compile(r"wall-clock time|rapid_p3_[A-Za-z0-9_]+")


def _run(case):
    import yaml

    case = dict(case)
    base = case.pop("base")
    if case.get("batch") is None:
        with open(base) as f:
            mp = yaml.safe_load(f)["model_param"]
        case["batch"] = mp["global_batch_size"]
        case["output"] = mp["decode_len"]
        case["prompt"] = mp["seq_len"] - mp["decode_len"]
    cfg = make_model_config(base, **case)
    res = run_config(cfg)
    lines = [ln for ln in res.stdout.splitlines() if not _DROP.search(ln)]
    return {"stdout": lines, "peak_gib": res.peak_gib}


def main(mode):
    if mode == "record":
        data = {}
        for name, case in CASES.items():
            data[name] = _run(case)
            print(f"recorded {name}")
        with open(BASELINE, "w") as f:
            json.dump(data, f, indent=1)
        print(f"baseline written to {BASELINE}")
        return 0
    with open(BASELINE) as f:
        data = json.load(f)
    bad = 0
    for name, case in CASES.items():
        got = _run(case)
        if got != data[name]:
            bad += 1
            exp = data[name]["stdout"]
            diff = [f"  - {a}\n  + {b}" for a, b in zip(exp, got["stdout"]) if a != b][:5]
            print(f"FAIL {name}\n" + "\n".join(diff))
        else:
            print(f"ok   {name}")
    print("all match" if not bad else f"{bad} case(s) changed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "check"))
