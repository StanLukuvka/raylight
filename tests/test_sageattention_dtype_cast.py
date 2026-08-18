"""Contract tests for the SageAttention dtype-cast bodge in attention.py.

The INT8-quantized H3 model computes attention in fp32 but SageAttention
kernels require fp16/bf16 inputs.  The dispatch wrapper must cast q/k/v to
fp16 (and join tensors) before the ring attention and cast the output back
to the model dtype -- and must do nothing for non-SAGE attention types.
"""

import pytest

import torch

from raylight.distributed_modules.attention import make_xfuser_attention


class _FakeRingAttention:
    """Stand-in for xFuserLongContextAttention recording the dt-hapes it saw."""

    seen = []

    def __call__(self, *args, **kwargs):
        query = args[1]  # xfuser call: (None, query, key, value, ...)
        self.seen.append(
            {
                "query": query.clone(),
                "key": args[2].clone(),
                "value": args[3].clone(),
                "join_q": kwargs.get("joint_tensor_query"),
            }
        )
        return query


@pytest.fixture
def fake_attn(monkeypatch):
    import raylight.distributed_modules.attention as mod

    fake = _FakeRingAttention()
    fake.seen.clear()
    monkeypatch.setattr(mod, "xFuserLongContextAttention", lambda **kw: fake)
    return fake


def _qkv(b, s, h, d):
    return (
        torch.randn(b, s, h * d, dtype=torch.float32),
        torch.randn(b, s, h * d, dtype=torch.float32),
        torch.randn(b, s, h * d, dtype=torch.float32),
    )


def test_sage_fp16_casts_fp32_qkv_to_half_and_restores_output(fake_attn):
    fn = make_xfuser_attention("SAGE_FP16", False)
    q, k, v = _qkv(1, 32, 8, 64)
    out = fn(q, k, v, heads=8, scale=0.125)
    seen = fake_attn.seen[-1]
    assert seen["query"].dtype == torch.float16
    assert seen["key"].dtype == torch.float16
    assert seen["value"].dtype == torch.float16
    assert out.dtype == torch.float32  # restored to the model dtype


def test_sage_fp16_casts_join_tensors(fake_attn):
    fn = make_xfuser_attention("SAGE_FP16", False)
    q, k, v = _qkv(1, 32, 8, 64)
    jq, jk, jv = _qkv(1, 16, 8, 64)
    out = fn(q, k, v, heads=8, join_q=jq, join_k=jk, join_v=jv, scale=0.125)
    seen = fake_attn.seen[-1]
    assert seen["join_q"].dtype == torch.float16
    assert out.dtype == torch.float32


def test_non_sage_attention_passes_fp32_through(fake_attn):
    fn = make_xfuser_attention("TORCH_EFFICIENT", False)
    q, k, v = _qkv(1, 32, 8, 64)
    out = fn(q, k, v, heads=8, scale=0.125)
    seen = fake_attn.seen[-1]
    assert seen["query"].dtype == torch.float32
    assert seen["key"].dtype == torch.float32
    assert out.dtype == torch.float32