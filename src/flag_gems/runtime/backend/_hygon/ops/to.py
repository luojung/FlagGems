# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dense FP32 to BF16 casts that preserve physical element order."""

import torch
import triton
import triton.language as tl

from flag_gems.ops.to import to_copy as default_to_copy
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def cast_dense(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, value.to(Y.dtype.element_ty), offsets < N)


def to_copy(
    x,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    target_device = x.device if device is None else torch.device(device)
    dense_format = memory_format in (None, torch.preserve_format) or (
        memory_format == torch.contiguous_format and x.is_contiguous()
    )
    if (
        x.dtype == torch.float32
        and dtype == torch.bfloat16
        and x.layout == torch.strided
        and layout in (None, torch.strided)
        and x.device.type in ("cuda", "flagos")
        and target_device == x.device
        and pin_memory is None
        and dense_format
        and not x.is_neg()
        and not x.is_conj()
        and 0 < x.numel() <= 2147483647
        and (x.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(x))
    ):
        # Dense permutations have no storage holes. Casting their physical order
        # into an output with identical strides preserves logical coordinates.
        with torch_device_fn.device(x.device):
            out = torch.empty_strided(x.shape, x.stride(), dtype=dtype, device=x.device)
            n = x.numel()
            block = 256 if n <= 4096 else 1024
            cast_dense[(triton.cdiv(n, block),)](
                x,
                out,
                n,
                BLOCK=block,
                num_warps=1,
                num_stages=1,
                allow_flush_denorm=False,
            )
        return out
    return default_to_copy(
        x,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        memory_format=memory_format,
    )
