# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native-lane FP64 MMAC regressions, including strided inputs and offsets."""

import pytest
import torch

import flag_gems

CASES = [
    (68, (1344, 256), (256, 1), 0, (256, 128), (128, 1), 131072),
    (74, (192, 256), (5120, 1), 0, (256, 128), (128, 1), 0),
    (75, (192, 89), (89, 1), 0, (89, 128), (128, 1), 0),
    (79, (192, 89), (89, 1), 0, (89, 6528), (6528, 1), 0),
    (80, (192, 89), (89, 1), 0, (89, 768), (768, 1), 0),
    (83, (2208, 64), (64, 1), 0, (64, 1280), (1280, 1), 0),
    (84, (2208, 64), (64, 1), 0, (64, 512), (512, 1), 0),
    (85, (2208, 64), (64, 1), 0, (64, 64), (64, 1), 0),
    (89, (24576, 368), (368, 1), 0, (368, 256), (256, 1), 0),
    (95, (576, 128), (128, 1), 0, (128, 128), (128, 1), 16384),
    (96, (576, 384), (384, 1), 0, (384, 128), (128, 1), 32768),
    (97, (960, 128), (128, 1), 0, (128, 128), (128, 1), 32768),
]


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon MMAC regression")
@pytest.mark.parametrize("seed", [0, 20261109, 20261110])
@pytest.mark.parametrize(
    "row, ashape, astride, aoffset, bshape, bstride, boffset", CASES
)
def test_hygon_mm_fp64_native_lane_regression(
    seed, row, ashape, astride, aoffset, bshape, bstride, boffset
):
    generator = torch.Generator().manual_seed(seed)
    inputs = []
    for shape, stride, offset in [
        (ashape, astride, aoffset),
        (bshape, bstride, boffset),
    ]:
        span = offset + 1 + sum((n - 1) * s for n, s in zip(shape, stride))
        backing = torch.randn(span, generator=generator, dtype=torch.float64)
        inputs.append(backing.as_strided(shape, stride, offset))
    a, b = inputs
    # Copy the complete backing so non-contiguous stride and nonzero offset survive.
    device_inputs = []
    for tensor in inputs:
        storage = tensor.as_strided(
            (tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,), 0
        )
        device_inputs.append(
            storage.to(flag_gems.device).as_strided(
                tensor.shape, tensor.stride(), tensor.storage_offset()
            )
        )
    with torch.no_grad():
        actual = flag_gems.mm(*device_inputs).cpu()
        reference = torch.mm(a, b)
    torch.testing.assert_close(actual, reference)
