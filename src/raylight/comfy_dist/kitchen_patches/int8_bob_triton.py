"""Research-only Triton W8A8 backend adapted from ComfyUI-INT8-Fast.

Source: BobJohnson24/ComfyUI-INT8-Fast@48a88b2fde88e986c6444fa1f51589b6089d04f3
The donor is AGPL-3.0. This module is an optional backend and is not used unless
RAYLIGHT_INT8_BACKEND=bob_triton.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_matmul_dequant_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    m_size,
    n_size,
    k_size,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    PER_CHANNEL_SCALE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m_size, BLOCK_M)
    num_pid_n = tl.cdiv(n_size, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_offset in range(0, tl.cdiv(k_size, BLOCK_K)):
        remaining = k_size - k_offset * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < m_size) & (offs_k[None, :] < remaining),
            other=0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < remaining) & (offs_n[None, :] < n_size),
            other=0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    activation_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < m_size, other=0.0)
    if PER_CHANNEL_SCALE:
        weight_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < n_size, other=0.0)
        total_scale = activation_scale[:, None] * weight_scale[None, :]
    else:
        weight_scale = tl.load(b_scale_ptr)
        total_scale = activation_scale[:, None] * weight_scale
    output = accumulator.to(tl.float32) * total_scale
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_size, other=0.0)
        output += bias[None, :]

    output_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    output_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    tl.store(output_ptrs, output, mask=output_mask)


def bob_triton_int8_linear(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Run Bob's tiled INT8 GEMM/dequant epilogue on one FSDP-materialized layer."""
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("bob_triton INT8 requires CUDA tensors")
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    weight = weight.contiguous()
    if x_2d.shape[1] != weight.shape[1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x_2d.shape[1]} and {weight.shape[1]}"
        )

    # The donor's one-program-per-row quantizer explicitly targets K <= 8192.
    # H3 includes K=14336, so retain Comfy Kitchen's proven row quantizer and
    # replace only the expensive GEMM + dequantization epilogue.
    from comfy_kitchen.backends.eager import quantization as eager_quantization

    x_int8, x_scale = eager_quantization.quantize_int8_rowwise(x_2d)
    m_size, k_size = x_int8.shape
    n_size = weight.shape[0]
    output = torch.empty((m_size, n_size), device=x.device, dtype=out_dtype)
    scales = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
    if scales.numel() not in (1, n_size):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {scales.numel()} values for {n_size} outputs"
        )
    bias = None if bias is None else bias.to(device=x.device, dtype=out_dtype).contiguous()

    block_m, block_n, block_k = 64, 128, 32
    grid = (triton.cdiv(m_size, block_m) * triton.cdiv(n_size, block_n),)
    _int8_matmul_dequant_kernel[grid](
        x_int8,
        weight,
        output,
        x_scale.reshape(-1).contiguous(),
        scales,
        x if bias is None else bias,
        m_size,
        n_size,
        k_size,
        x_int8.stride(0),
        x_int8.stride(1),
        weight.stride(1),
        weight.stride(0),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=8,
        HAS_BIAS=bias is not None,
        PER_CHANNEL_SCALE=scales.numel() != 1,
        num_warps=4,
        num_stages=3,
    )
    return output.reshape(*original_shape[:-1], n_size)
