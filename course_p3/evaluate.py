"""Project 3 evaluations: the required grid plus case studies.

    python course_p3/evaluate.py required     # PROJECTS.md evaluation
    python course_p3/evaluate.py sweep        # slides: S x B x (b, R) grid vs AR
    python course_p3/evaluate.py crossover    # where AR catches up, A100 vs H100
    python course_p3/evaluate.py head         # sensitivity: decode output head priced for all B*q rows
    python course_p3/evaluate.py all

Results are written as CSV to course_p3/results/.
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runner import HW_A100, LLAMA8B, REPO, make_model_config, run_config  # noqa: E402

HW_H100 = os.path.join(REPO, "configs/hardware-config/course_project/H100_SXM5_80GB.yaml")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
WORKERS = int(os.environ.get("P3_WORKERS", "8"))

# Llama-3.1-8B shape (for the analytic FLOP count only; timing comes from RAPID-LLM)
H, L, I, V, NH, KVH = 4096, 32, 14336, 128256, 32, 8
HD = H // NH
LAYER_PARAMS = H * (H + 2 * KVH * HD) + H * H + 3 * H * I


def call_flops(batch, q, ctx):
    """Forward FLOPs of one decode call with q positions per sequence attending to ctx keys."""
    dense = 2 * batch * q * (L * LAYER_PARAMS + H * V)
    attn = 2 * 2 * batch * q * ctx * NH * HD * L
    return dense + attn


def decode_flops(batch, prompt, output, b=None, rounds=None, kv_save=False):
    if b is None:
        return sum(call_flops(batch, 1, prompt + t + 1) for t in range(output))
    cpb = rounds + (1 if kv_save else 0)
    return cpb * sum(call_flops(batch, b, prompt + (k + 1) * b) for k in range(output // b))


def diff_param(b, r, kv_save=False):
    return {"decode_schedule": "block_diffusion", "block_size": b, "refine_rounds": r, "kv_save_pass": kv_save}


def one(job):
    hw = job.get("hw", HW_A100)
    pp = diff_param(job["b"], job["R"], job.get("kv_save", False)) if job.get("b") else {}
    if job.get("head_all_rows"):
        pp["decode_head_all_rows"] = True
    cfg = make_model_config(LLAMA8B, batch=job["B"], prompt=job["S"], output=job["O"], project_param=pp or None)
    res = run_config(cfg, hw)
    calls = res.model_calls if res.model_calls is not None else job["O"]
    flops = decode_flops(job["B"], job["S"], job["O"], job.get("b"), job.get("R"), job.get("kv_save", False))
    row = dict(job)
    row["hw"] = os.path.basename(hw).replace(".yaml", "")
    row.update(
        schedule="block_diffusion" if job.get("b") else "AR",
        prefill_s=res.prefill_s,
        decode_s=res.decode_s,
        total_s=res.total_s,
        ttft_s=res.ttft_s,
        model_calls=calls,
        ms_per_call=1e3 * res.decode_s / calls,
        decode_tok_per_s=job["B"] * job["O"] / res.decode_s,
        decode_tflop=flops / 1e12,
        peak_gib=res.peak_gib,
        prefill_peak_gib=res.prefill_peak_gib,
        decode_peak_gib=res.decode_peak_gib,
        fits=res.peak_gib <= 80.0,
    )
    return row


def run_jobs(jobs, name):
    with ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(one, jobs))
    os.makedirs(OUT, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path = os.path.join(OUT, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {path}")
    return rows


def required():
    jobs = []
    for B in (1, 16, 64):
        for O in (128, 256):
            base = dict(B=B, S=512, O=O)
            jobs.append(dict(base))
            jobs.append(dict(base, b=8, R=4))
            jobs.append(dict(base, b=8, R=4, kv_save=True))
    return run_jobs(jobs, "required")


def max_batch(S, O, hw=HW_A100):
    """Largest batch whose AR peak memory fits in 80 GiB (doubling, then bisection)."""
    def fits(B):
        cfg = make_model_config(LLAMA8B, batch=B, prompt=S, output=O)
        return run_config(cfg, hw).peak_gib <= 80.0

    lo, hi = 1, 2
    while fits(hi):
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


BR = [(4, 2), (8, 2), (8, 4), (16, 2), (16, 4), (16, 8), (32, 2), (32, 4), (32, 8)]


def sweep():
    jobs = []
    for S in (2048, 8192, 32768):
        bmax = max_batch(S, 128)
        print(f"S={S}: largest batch that fits = {bmax}")
        for B in sorted({b for b in (1, 8, 32, 64) if b <= bmax} | {bmax}):
            jobs.append(dict(S=S, O=128, B=B, Bmax=bmax))
            for b, r in BR:
                jobs.append(dict(S=S, O=128, B=B, Bmax=bmax, b=b, R=r))
    return run_jobs(jobs, "sweep")


def crossover():
    batches = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    jobs = []
    for hw in (HW_A100, HW_H100):
        for S in (512, 8192):
            for B in batches:
                jobs.append(dict(hw=hw, S=S, O=128, B=B))
                for b, r in ((8, 4), (16, 4), (32, 8), (8, 2)):
                    jobs.append(dict(hw=hw, S=S, O=128, B=B, b=b, R=r))
    return run_jobs(jobs, "crossover")


def head():
    """Same runs with the decode output head priced for B*q rows vs RAPID-LLM's default (q rows)."""
    jobs = []
    for S, batches in ((512, (1, 16, 64, 128, 256)), (8192, (1, 8, 16))):
        for B in batches:
            for flag in (False, True):
                base = dict(S=S, O=128, B=B, head_all_rows=flag)
                jobs.append(dict(base))
                jobs.append(dict(base, b=8, R=4))
                jobs.append(dict(base, b=16, R=4))
    return run_jobs(jobs, "head_sensitivity")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "required"
    for name in (("required", "sweep", "crossover", "head") if what == "all" else (what,)):
        globals()[name]()
