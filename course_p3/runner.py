"""Run RAPID-LLM from Python with generated model configs (Project 3 helpers).

Every run goes through the normal command line (run_perf.py) in a subprocess, so the
numbers are exactly what a student would see by hand.
"""

from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW_A100 = os.path.join(REPO, "configs/hardware-config/course_project/a100_80GB_1gpu.yaml")
LLAMA8B = os.path.join(REPO, "configs/model-config/course_project/Llama3.1-8B_inf.yaml")


@dataclass
class RunResult:
    prefill_s: float
    decode_s: float
    total_s: float
    ttft_s: float
    peak_gib: float
    prefill_peak_gib: float
    decode_peak_gib: float
    model_calls: Optional[int] = None
    stdout: str = ""


def make_model_config(
    base: str = LLAMA8B,
    *,
    batch: int,
    prompt: int,
    output: int,
    project_param: Optional[dict] = None,
    sample_every: Optional[int] = None,
    drop_window: bool = False,
) -> dict:
    with open(base, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    mp = cfg["model_param"]
    mp["global_batch_size"] = int(batch)
    mp["seq_len"] = int(prompt) + int(output)
    mp["decode_len"] = int(output)
    if drop_window:
        mp.get("attention", {}).pop("window", None)
    if sample_every is not None:
        cfg.setdefault("inference_param", {})["sample_every"] = int(sample_every)
    if project_param:
        cfg["project_param"] = dict(project_param)
    return cfg


_FLOAT = r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"


def _grab(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def run_config(cfg: dict, hw: str = HW_A100, *, workdir: Optional[str] = None) -> RunResult:
    """Run one config. Without ``workdir`` it runs in a temporary folder that is removed after."""
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="rapid_p3_") as tmp:
            return run_config(cfg, hw, workdir=tmp)
    model_path = os.path.join(workdir, "model.yaml")
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "run_perf.py"), "--hardware_config", hw, "--model_config", model_path],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"run_perf failed ({proc.returncode}):\n{out[-4000:]}")

    with open(os.path.join(workdir, "output/LLM/memory-summary/memory_capacity_comparison.txt"), encoding="utf-8") as f:
        mem_text = f.read()
    with open(os.path.join(workdir, "output/LLM/LLM_inference_results.txt"), encoding="utf-8") as f:
        results_text = f.read()

    prefill = _grab(r"\[prefill\] time: " + _FLOAT, out)
    decode = _grab(r"\[decode\] time: " + _FLOAT, out)
    total = _grab(r"\[total\] time: " + _FLOAT, out)
    calls = _grab(r"\[diffusion\] model_calls=([0-9]+)", out)
    return RunResult(
        prefill_s=prefill,
        decode_s=decode,
        total_s=total,
        ttft_s=_grab(r"Time to First Token: " + _FLOAT, results_text),
        peak_gib=_grab(r"Max peak memory usage \(per gpu\): " + _FLOAT, mem_text),
        prefill_peak_gib=_grab(r"Prefill peak memory usage \(per gpu\): " + _FLOAT, mem_text),
        decode_peak_gib=_grab(r"Final decode peak memory usage \(per gpu\): " + _FLOAT, mem_text),
        model_calls=int(calls) if calls is not None else None,
        stdout=out,
    )
