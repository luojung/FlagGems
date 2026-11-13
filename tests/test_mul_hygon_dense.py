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

SHAPES = [
    ((256, 256), (1, 256)),
    ((1, 128, 512), (65536, 1, 128)),
    ((1, 256, 4096), (1048576, 1, 256)),
    ((1, 362, 5793), (1048576, 1, 362)),
    ((4, 128, 128), (1, 4, 512)),
    ((2, 1, 128, 256), (1, 65536, 2, 256)),
]


def make_pair(shape, stride, offset, shared):
    span = 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    g = torch.Generator().manual_seed(20261113)
    if shared:
        backing = [torch.randn(2 * span + offset + 19, generator=g).bfloat16()]
        args = [
            backing[0].as_strided(shape, stride, offset),
            backing[0].as_strided(shape, stride, offset + span),
        ]
        moved = [backing[0].to(flag_gems.device)]
        device = [
            moved[0].as_strided(shape, stride, offset),
            moved[0].as_strided(shape, stride, offset + span),
        ]
    else:
        backing = [
            torch.randn(span + offset + 19, generator=g).bfloat16() for _ in range(2)
        ]
        args = [t.as_strided(shape, stride, offset) for t in backing]
        moved = [t.to(flag_gems.device) for t in backing]
        device = [t.as_strided(shape, stride, offset) for t in moved]
    return backing, args, moved, device


@pytest.mark.mul
@pytest.mark.parametrize("shape,stride", SHAPES)
@pytest.mark.parametrize("offset", [0, 7])
@pytest.mark.parametrize("shared", [False, True])
def test_mul_hygon_dense_layout(shape, stride, offset, shared):
    backing, args, moved, device = make_pair(shape, stride, offset, shared)
    expected = torch.ops.aten.mul.Tensor(*args)
    actual = flag_gems.mul(*device)
    torch.testing.assert_close(actual.cpu(), expected)
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert not any(torch._C._is_alias_of(actual, t) for t in device)
    again = flag_gems.mul(*device)
    assert not torch._C._is_alias_of(actual, again)
    assert torch._C._is_alias_of(*device) == shared
    for a, b in zip(backing, moved):
        assert torch.equal(a.view(torch.uint8), b.cpu().view(torch.uint8))


@pytest.mark.mul
@pytest.mark.parametrize(
    "shape,stride",
    [
        ((255, 256), (1, 255)),
        ((257, 256), (1, 257)),
        ((2048, 2048), (1, 2048)),
        ((2049, 2048), (1, 2049)),
        ((256, 256), (1, 257)),
        ((256, 256), (0, 1)),
    ],
)
def test_mul_hygon_dense_boundaries(shape, stride):
    backing, args, moved, device = make_pair(shape, stride, 7, False)
    expected = torch.ops.aten.mul.Tensor(*args)
    actual = flag_gems.mul(*device)
    torch.testing.assert_close(actual.cpu(), expected)
    assert actual.stride() == expected.stride()
    for a, b in zip(backing, moved):
        assert torch.equal(a.view(torch.uint8), b.cpu().view(torch.uint8))


@pytest.mark.mul
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_mul_hygon_dense_fallback_out_and_dtype(dtype):
    a = (
        torch.randn((256, 256), generator=torch.Generator().manual_seed(88))
        .to(dtype)
        .t()
    )
    b = torch.randn((256, 256), generator=torch.Generator().manual_seed(89)).to(dtype)
    expected = a * b
    da, db = a.to(flag_gems.device), b.to(flag_gems.device)
    actual = flag_gems.mul(da, db)
    torch.testing.assert_close(actual.cpu(), expected)
    out = torch.empty_like(da)
    returned = flag_gems.mul(da, db, out=out)
    assert returned is out
    torch.testing.assert_close(out.cpu(), expected)
    flag_gems.mul_(da, db)
    torch.testing.assert_close(da.cpu(), expected)


@pytest.mark.mul
def test_mul_hygon_dense_nonfinite_and_same_object():
    storage = torch.arange(65536, dtype=torch.float32).bfloat16()
    storage[:6] = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0],
        dtype=torch.bfloat16,
    )
    cpu = storage.as_strided((256, 256), (1, 256))
    device = storage.to(flag_gems.device).as_strided((256, 256), (1, 256))
    actual = flag_gems.mul(device, device)
    torch.testing.assert_close(actual.cpu(), cpu * cpu, equal_nan=True)
    assert torch.equal(device.cpu().view(torch.int16), cpu.view(torch.int16))


@pytest.mark.mul
@pytest.mark.parametrize(
    "shape,stride",
    [((257, 257), (1, 257)), ((513, 129), (1, 513)), ((362, 5793), (1, 362))],
)
@pytest.mark.parametrize("offset", [0, 1, 7])
def test_mul_hygon_dense_masked_tail(shape, stride, offset):
    backing, args, moved, device = make_pair(shape, stride, offset, True)
    expected = torch.ops.aten.mul.Tensor(*args)
    actual = flag_gems.mul(*device)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert not any(torch._C._is_alias_of(actual, t) for t in device)
    for original, current in zip(backing, moved):
        assert torch.equal(original.view(torch.uint8), current.cpu().view(torch.uint8))


@pytest.mark.mul
def test_mul_hygon_dense_tail_nonfinite():
    backing, args, moved, device = make_pair((257, 257), (1, 257), 1, True)
    for storage in backing:
        storage[-270:-264] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0],
            dtype=torch.bfloat16,
        )
    moved[0].copy_(backing[0].to(flag_gems.device))
    expected = args[0] * args[1]
    actual = flag_gems.mul(*device).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = ~torch.isnan(expected)
    assert torch.equal(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )
