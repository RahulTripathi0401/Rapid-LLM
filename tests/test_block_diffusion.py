"""Project 3: block-diffusion decode (query_tokens > 1) and its schedule."""

from types import SimpleNamespace

import pytest

import llm_util
from simulate_inference_graph import block_diffusion_schedule, project_flag


def _ctx():
    return SimpleNamespace(use_moe=False, moe_num_experts=1, attention_type="gqa", run_type="inference", head_dim=None)


def _shapes(q, ctx_len=600):
    return llm_util.autoregressive_decoder_gemm(
        _ctx(), batch_size=4, current_seq_len=ctx_len, d_model=4096, num_heads=32, kv_heads=8,
        intermediate_size=14336, vocab_size=128256, model_type="llama", query_tokens=q,
    )


def test_q1_shapes_are_single_token():
    s = _shapes(1)
    assert s["qkv_proj"] == (4, 1, 4096, 4096 + 2 * 1024)
    assert s["attention_score"] == (4 * 8, 4, 128, 600)
    assert s["ffn1"][1] == 1 and s["linear"][1] == 1


def test_q_scales_rows_not_weights_or_context():
    one, eight = _shapes(1), _shapes(8)
    for key in ("qkv_proj", "output_proj", "ffn1", "ffn2", "linear"):
        b1, m1, k1, n1 = one[key]
        b8, m8, k8, n8 = eight[key]
        assert (b8, m8, k8, n8) == (b1, 8 * m1, k1, n1), key  # same weights, 8x rows
    # every query attends to the same keys: K/V read once per call
    assert eight["attention_score"] == (32, 8 * 4, 128, 600)
    assert eight["attention_output"] == (32, 8 * 4, 600, 128)


def test_query_tokens_must_be_positive():
    with pytest.raises(ValueError):
        _shapes(0)


def test_schedule_defaults_to_autoregressive():
    assert block_diffusion_schedule({}, 128) is None
    assert block_diffusion_schedule({"decode_schedule": "autoregressive"}, 128) is None


def test_schedule_counts():
    s = block_diffusion_schedule({"decode_schedule": "block_diffusion", "block_size": 8, "refine_rounds": 4}, 128)
    assert (s.num_blocks, s.calls_per_block, s.model_calls) == (16, 4, 64)
    assert s.context_len(512, 0) == 520 and s.context_len(512, 15) == 640
    s2 = block_diffusion_schedule(
        {"decode_schedule": "block_diffusion", "block_size": 8, "refine_rounds": 4, "kv_save_pass": True}, 128
    )
    assert s2.model_calls == 80  # slides' example: R + 1 calls per block


def test_neutral_schedule_matches_token_loop():
    s = block_diffusion_schedule({"decode_schedule": "block_diffusion", "block_size": 1, "refine_rounds": 1}, 256)
    assert s.model_calls == 256
    assert [s.context_len(512, k) for k in (0, 255)] == [513, 768]  # same as AR steps 0 and 255


@pytest.mark.parametrize(
    "param",
    [
        {"decode_schedule": "block_diffusion", "block_size": 3},  # 128 not a multiple of 3
        {"decode_schedule": "block_diffusion", "block_size": 8, "refine_rounds": 0},
        {"decode_schedule": "block_diffusion", "blocksize": 8},  # typo
        {"decode_schedule": "diffusion"},
        {"decode_schedule": "block_diffusion", "block_size": 8, "kv_save_pass": "false"},  # quoted bool
        {"decode_head_all_row": True},  # typo, even with the default AR schedule
    ],
)
def test_schedule_rejects_bad_input(param):
    with pytest.raises(ValueError):
        block_diffusion_schedule(param, 128)


def test_project_flag_is_strict():
    assert project_flag({}, "decode_head_all_rows") is False
    assert project_flag({"decode_head_all_rows": True}, "decode_head_all_rows") is True
    with pytest.raises(ValueError):
        project_flag({"decode_head_all_rows": "yes"}, "decode_head_all_rows")


def test_head_flag_alone_keeps_ar_schedule():
    assert block_diffusion_schedule({"decode_head_all_rows": True}, 128) is None
