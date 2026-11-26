"""Probe RAPID-LLM's pricing of the decode attention GEMMs as the query rows grow.

Shows the tiler cliff at 128 query rows (presentation slide 11, report §6):
    python course_p3/tiler_probe.py      -> course_p3/results/tiler_probe.csv
"""

import csv
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import yaml  # noqa: E402

import config  # noqa: E402
from inference_timing import TimeCalculationLLMInference  # noqa: E402
from runner import HW_A100, make_model_config  # noqa: E402
from train_timing import GemmType  # noqa: E402

BATCH, KV_HEADS, HEAD_DIM, CONTEXT = 64, 8, 128, 8192
PEAK_FLOPS = 312e12  # A100 BF16, the hardware config's nominal rate
ROWS = (4, 8, 16, 32, 64, 96, 112, 120, 128, 136, 160, 192, 256)


def main():
    cfg = make_model_config(batch=BATCH, prompt=CONTEXT, output=128)
    out_path = os.path.join(HERE, "results", "tiler_probe.csv")
    with tempfile.TemporaryDirectory(prefix="rapid_p3_probe_") as tmp:
        model_path = os.path.join(tmp, "model.yaml")
        with open(model_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        hw = config.parse_config(HW_A100, config_type="hardware")
        mc = config.parse_config(model_path, config_type="LLM")
        tc = TimeCalculationLLMInference(hw, mc, "LLM", output_dir=os.path.join(tmp, "out"))
        rows = []
        for m in ROWS:
            score = tc.parallelism_gemm_forward((BATCH * KV_HEADS, m, HEAD_DIM, CONTEXT), "probe_score",
                                                gemm_type=GemmType.ATTENTION_SCORE, decode=True)
            output = tc.parallelism_gemm_forward((BATCH * KV_HEADS, m, CONTEXT, HEAD_DIM), "probe_output",
                                                 gemm_type=GemmType.ATTENTION_OUTPUT, decode=True)
            flops = score[3] + output[3]
            rows.append({"query_rows": m, "attention_score_ms": round(score[0] * 1e3, 3),
                         "attention_output_ms": round(output[0] * 1e3, 3),
                         "ideal_compute_ms": round(flops / PEAK_FLOPS * 1e3, 3)})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
