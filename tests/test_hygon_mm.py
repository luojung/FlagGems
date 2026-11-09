# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.mm,
    pytest.mark.skipif(
        flag_gems.vendor_name != "hygon", reason="Hygon GEMM regression coverage"
    ),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("m,n,k", [(0, 17, 9), (13, 0, 9), (13, 17, 0)])
def test_hygon_mm_empty(dtype, m, n, k):
    a = torch.empty((m, k), dtype=dtype, device=flag_gems.device)
    b = torch.empty((k, n), dtype=dtype, device=flag_gems.device)
    y = flag_gems.mm(a, b)
    assert y.shape == (m, n)
    assert y.dtype == dtype
    if k == 0:
        assert torch.count_nonzero(y.cpu()).item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [7, 101])
def test_hygon_mm_offset_and_both_strides(dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    # Offset, padded rows and a column stride greater than one on both inputs.
    a = torch.randn(41 * 137 + 11, dtype=dtype, generator=gen).as_strided(
        (33, 65), (137, 2), 7
    )
    b = torch.randn(65 * 83 + 11, dtype=dtype, generator=gen).as_strided(
        (65, 37), (83, 2), 5
    )

    # Transfer underlying storage to preserve the requested offset and strides.
    def transfer(x):
        storage = torch.as_strided(
            x, (x.untyped_storage().nbytes() // x.element_size(),), (1,), 0
        )
        return storage.to(flag_gems.device).as_strided(
            x.shape, x.stride(), x.storage_offset()
        )

    aa, bb = transfer(a), transfer(b)
    y = flag_gems.mm(aa, bb).cpu().double()
    ref = a.double() @ b.double()
    tol = 1e-10 if dtype == torch.float64 else 2e-4
    torch.testing.assert_close(y, ref, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (192, 128, 128),
        (192, 6528, 89),
        (2208, 1280, 64),
        (1344, 128, 256),
        (2208, 64, 64),
        (576, 128, 128),
        (2208, 512, 64),
        (576, 128, 384),
    ],
)
def test_hygon_mm_fp64_keeps_low_mantissa_bits(m, n, k):
    # These bits disappear if an MMAC operand is silently converted to FP32.
    a = torch.zeros((m, k), dtype=torch.float64)
    b = torch.ones((k, n), dtype=torch.float64)
    a[:, 0] = 1.0 + 2.0**-30
    a[:, 1] = -1.0
    y = flag_gems.mm(a.to(flag_gems.device), b.to(flag_gems.device)).cpu()
    torch.testing.assert_close(y, a @ b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1344, 128, 256),
        (2208, 64, 64),
        (576, 128, 128),
        (2208, 512, 64),
        (576, 128, 384),
    ],
)
@pytest.mark.parametrize("seed", [1238, 101241, 907])
def test_hygon_mm_fp64_reported_peel_regression(m, n, k, seed):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.float64, generator=gen)
    b = torch.randn((k, n), dtype=torch.float64, generator=gen)
    reference = a @ b
    aa = a.to(flag_gems.device)
    outputs = []
    for offset in [0, 131072]:
        # Preserve identical logical inputs while changing only B's offset.
        storage = torch.empty(offset + b.numel(), dtype=b.dtype, device=aa.device)
        bb = storage.as_strided(b.shape, b.stride(), offset)
        bb.copy_(b)
        assert bb.storage_offset() == offset
        assert torch.equal(bb.cpu(), b)
        result = flag_gems.mm(aa, bb).cpu()
        torch.testing.assert_close(result, reference, rtol=1e-10, atol=1e-10)
        outputs.append(result)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
