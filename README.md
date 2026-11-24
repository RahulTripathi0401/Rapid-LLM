# ECE201C Project 3: block-diffusion decoding in RAPID-LLM

This fork adds **block-diffusion LLM decoding** to RAPID-LLM and compares it with normal
autoregressive (AR) decoding on Llama-3.1-8B and one A100 80 GB (all numbers are RAPID-LLM's
analytical estimates; no GPU is needed). The full write-up is
[course_p3/REPORT.md](course_p3/REPORT.md). The original course README follows the project
section below.

**What is modeled.** The output is written in blocks of `b` tokens. Each block is refined over `R`
model calls; every call processes all `b` positions at once and attends to the KV cache of all
earlier text plus the whole block. So a call reads the weights and KV cache once but does `b`
positions of work, and each output token costs `R` position-passes instead of 1.

### Results (required evaluation: prompt 512, output 128, b = 8, R = 4, A100)

| Batch | AR total | Block diffusion total | Speedup | Model calls (AR → diffusion) | Peak memory |
|---|---|---|---|---|---|
| 1 | 1.586 s | 0.815 s | 1.95× | 128 → 64 | 15.14 GiB (both) |
| 16 | 2.191 s | 1.501 s | 1.46× | 128 → 64 | 17.91 GiB (both) |
| 64 | 4.131 s | 4.433 s | 0.93× (AR wins) | 128 → 64 | 26.79 GiB (both) |

Output 256 gives the same ratios (1.97×, 1.54×, 0.92×). With one extra K/V-saving call per block
(`kv_save_pass: true`, R + 1 = 5 calls): 1.57×, 1.24×, 0.81×.

**Main findings** (details, plots and the case studies in the report):
- While a decode call is memory-bound, `b` positions cost about the same as one token (12.16 vs
  12.11 ms at batch 1), so diffusion is about **b / R** times faster.
- When batch × b positions make a call compute-bound, the R× extra FLOPs win: for b = 8, R = 4 at
  a 512-token prompt AR catches up at **batch 64 on the A100** and **batch 192 on the H100**.
- Long prompts keep diffusion ahead (the KV cache read dominates and memory caps the batch):
  b = 16, R = 4 at the largest batch that fits is 1.07× at 2K, 1.94× at 8K and 2.62× at 32K.
- Peak memory and the batch that fits are unchanged; the first tokens arrive later.

### Running it

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.12
pip install -r requirements.txt pytest matplotlib
pytest -q                                              # 50 passed, 2 skipped

# one block-diffusion run (batch 1, prompt 512, output 128, b = 8, R = 4)
python run_perf.py --hardware_config configs/hardware-config/course_project/a100_80GB_1gpu.yaml \
                   --model_config configs/model-config/course_project/Llama3.1-8B_block_diffusion.yaml
```

The run prints a `[diffusion]` line with the model calls, blocks and time per call. To use block
diffusion with any model config, add this block next to `model_param` (course helper H1):

```yaml
project_param:
  decode_schedule: block_diffusion   # autoregressive (default) or block_diffusion
  block_size: 8                      # b; must divide decode_len
  refine_rounds: 4                   # R
  kv_save_pass: false                # true: R + 1 calls per block
  decode_head_all_rows: false        # true: sensitivity check, see "Notes" below
```

Reproduce every number in the report:

```bash
python course_p3/regress.py check          # feature off: 11 runs byte-identical to the original code
python course_p3/evaluate.py required      # ~1 min  -> course_p3/results/required.csv
python course_p3/evaluate.py sweep         # ~25 min -> results/sweep.csv (prompt 2K/8K/32K grid)
python course_p3/evaluate.py crossover     # ~25 min -> results/crossover.csv (A100 vs H100)
python course_p3/evaluate.py head          # ~2 min  -> results/head_sensitivity.csv
python course_p3/plots.py                  # report figures -> course_p3/figures/
python course_p3/slide_figs.py             # presentation figures -> course_p3/figures/slides/
```

`regress.py check` compares against `course_p3/baseline.json`, recorded before any code change
(the course's `course/regress.py` is not on this branch, so this is the equivalent).

### Code changes

| File | Change |
|---|---|
| `llm_util.py` | `autoregressive_decoder_gemm`: helper H2 `query_tokens` (q) is now used. Every per-token GEMM has B·q rows; attention has q query positions against the full context. |
| `inference_timing.py` | Decode layer timing uses q positions (norms, residuals, MoE tokens, output head); decode peak memory uses q = b; time to first token, tokens/s and KV-read bytes follow the block schedule; prints the `[diffusion]` summary. |
| `simulate_inference_graph.py` | `BlockDiffusionSchedule` and `block_diffusion_schedule()` parse and validate `project_param`; `_build_block_diffusion_decode` walks blocks instead of tokens (sampled blocks, trapezoid rule over the block index, `calls_per_block` calls each). |
| `tests/test_block_diffusion.py` | 14 tests: q = 1 shapes unchanged, q scales rows but not weights or context, call counts, the neutral schedule, input validation. |
| `course_p3/` | Regression check, run helpers, evaluation, plots, results (CSV), figures and the report. |

### Checks

- **Feature off:** `course_p3/regress.py check` — all 11 runs match the original outputs byte for byte.
- **Neutral setting:** b = 1, R = 1 reproduces normal decode exactly (decode time, prefill, time to
  first token, peak memory, tokens/s) at batch 1/64/8 with default and denser sampling.
- **Unit tests:** `pytest -q`.

### Notes and limitations

- **R vs R + 1 calls.** PROJECTS.md says a block is refined over R calls; the class slides add a
  call that saves the block's K/V. R is the default; `kv_save_pass: true` gives R + 1. Both are
  reported.
- **Decode output head.** RAPID-LLM prices the decode output-head matmul from a 4-D shape whose
  batch dimension `single_gpu_gemm_forward` ignores, so it is charged for q rows, not B·q (also
  raised on Piazza @54). The default is left unchanged so feature-off runs stay identical;
  `decode_head_all_rows: true` prices all B·q rows. It changes nothing at batch ≤ 8 and adds
  3–6% to diffusion decode time at batch 64–256 (≤ 1% for AR): the batch-64 required speedup goes
  from 0.93× to 0.91×. No conclusion changes (`course_p3/results/head_sensitivity_summary.csv`).
- **Tiler cliff at 128 query rows.** RAPID-LLM's GEMM tiler prices the decode attention-output GEMM
  ~14× higher at M ≥ 128 rows than at M = 120 for nearly the same work. Llama-8B (4 query heads per
  KV head) reaches M = 128 at b = 32, so b = 32 results are pessimistic and left out of the
  conclusions; b ≤ 16 and the required b = 8 runs are below it.
- Timing only: text quality and adaptive (confidence-based) refinement are not modeled; R is an
  input. `decode_len` must be a multiple of b.

---

# RAPID-LLM (course branch)

RAPID-LLM estimates how long a large language model takes to run on given hardware, and how much
GPU memory it needs. It does not run the model. It turns the model shape (layers, hidden size,
heads, experts) into matrix multiplications and other operations, prices each one from the GPU's
compute rate and memory bandwidth, adds communication time when several GPUs work together, and
adds everything up.

This course branch runs on the analytical backend. This file covers setup, running a config and
how RAPID-LLM prices a decode step. For the course projects, read
[COURSE_README.md](COURSE_README.md): the helpers we added and where to start in the code.

## Setup (Python 3.12)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest
pytest -q                           # should pass
```

## Running one config

```bash
python run_perf.py --hardware_config configs/hardware-config/course_project/a100_80GB_1gpu.yaml \
                   --model_config configs/model-config/course_project/Llama3.1-8B_inf.yaml
```

Results go to `output/LLM/`: `LLM_inference_results.txt` repeats the summary, and
`memory-summary/memory_capacity_comparison.txt` gives the prefill, decode and maximum peak memory
per GPU. The lines that matter on screen:

```
[decode] sample step 0: seq_len=1025, time=0.0121s
[decode] sample step 1023: seq_len=2048, time=0.0122s
[prefill] time: 0.0661s, [decode] time: 12.4618s, [total] time: 12.5279s
[kv-cache] prefill_store=0.1 GiB, decode_store=0.1 GiB, decode_fetch=192.0 GiB
LLM time to first token: 0.08s
Aggregate decode throughput tok/s (batch=1, replica_count=1): start=82.40, mid(token 512)=82.17, end=81.94
```

- `sample step`: one decode step (one new token for every sequence) at that context length.
- `prefill` processes the prompt once; `decode` generates all output tokens; time to first token
  is prefill plus the first decode step.
- `kv-cache`: key/value bytes written in prefill and decode and read in decode. `tok/s`: tokens
  per second over the whole batch at the start, middle and end of decoding.

## Configs

Put all your settings in the model config. The hardware configs are shared: do not edit them.

`configs/model-config/course_project/` holds the four project models, and
`configs/hardware-config/course_project/` the GPUs used in the course.

| Model config | What it is |
|---|---|
| `course_project/qwen3_30b_a3b_inf.yaml` | Qwen3-30B-A3B, mixture of experts (Project 1) |
| `course_project/gemma_3_4b_inf.yaml` | Gemma 3 4B text model with sliding window (Project 2) |
| `course_project/Llama3.1-8B_inf.yaml` | Llama-3.1-8B, dense (Project 3) |
| `course_project/mimo_7b_inf.yaml` | MiMo-7B base model, dense (Project 4) |
| `Llama3.1-70B_inf.yaml`, `Llama3.1-405B_inf.yaml` | larger dense Llama models |
| `DeepSeekV3_inf.yaml`, `GLM4.7_358B_inf.yaml` | large mixture-of-experts models |
| `minicpm3_4b_inf.yaml` | a small dense model with MLA attention |

| Hardware config | What it is |
|---|---|
| `course_project/a100_80GB_1gpu.yaml` | one A100 80 GB (all project evaluations) |
| `course_project/a100_80GB.yaml` | two A100 80 GB, tensor parallel (tp: 2) |
| `course_project/H100_SXM5_80GB.yaml` | one H100 SXM5 80 GB |
| `waferscale_20v100_80hbm.yaml` | 20 V100-class GPUs in one node, tensor parallel |

The Gemma file is refused until Project 2 is done; delete its `window:` block to run it as a dense
model. Units: in the YAML files `GB` means 2^30 bytes (as a bandwidth, 2^30 bytes per second);
latencies are in seconds.

## Changing the inputs

In the model YAML, under `model_param`: `global_batch_size` is the number of sequences served
together (B), `decode_len` the decode (output) tokens per sequence, and `seq_len` the prefill plus
decode tokens; to change only the output length, change both by the same amount. Under
`inference_param`, `sample_every: -1` (the default) prices only the first and last decode step;
`sample_every: N` also prices every Nth step.

Your own inputs (link bandwidth and latency, window size, block size, number of guesses, and so
on) go in a `project_param:` block in the model YAML; see H1 in [COURSE_README.md](COURSE_README.md).

## How a decode step is priced

1. `TimeCalculationLLMInference.calc_total_inference_time` (inference_timing.py) runs prefill
   and peak memory in `calc_time`, then decode in `calc_decode_time`, which hands over to
   `DecodeGraph.build_decode_graph` (simulate_inference_graph.py).
2. `build_decode_graph` picks the sampled steps (`_generate_sample_points`; by default only the
   first and last step), computes the GEMM shapes for the current context length
   (`llm_util.process_decode_gemm_shapes`, which calls `autoregressive_decoder_gemm`) and calls
   `_execute_decode_step`.
3. `_execute_decode_step` makes a fresh timing object and calls `prepare_decode_graphs`
   (inference_timing.py).
4. `prepare_decode_graphs` calls `_build_decode_transformer_results` for a dense layer and, for
   MoE models, an MoE layer. Each call returns a table of operations (`qkv_proj`, `attention`,
   `ffn1`, ...) and a `node_breakdown` holding the per-layer time `transformer_time_f` and the
   output-head time `linear_softmax_f`.
5. `_prepare_execution_graphs` (train_timing.py) turns these into a graph of layers
   (simulate_train_graph.py), and `LLMExecutionDispatcher.run` (llm_execution.py) returns the
   step time.
6. `_integrate_decode_samples` joins the sampled steps with straight lines (trapezoid rule) to
   get the total decode time.

Peak memory: `calc_time` calls `MemoryEstimator.build_memory_data` (memory_estimation.py) for
weights, KV cache and activations, then `simulate_peak`, which replays the layer graph
(simulate_train_graph.py). Going over the GPU's memory prints a warning; it does not stop the run.
