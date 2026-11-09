import pytest
import torch

import flag_gems


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon STATE layout")
@pytest.mark.parametrize(
    "scale,pattern",
    [
        (None, "random"),
        (128**-0.5, "random"),
        (0.125, "random"),
        (-(128**-0.5), "random"),
        (None, "mixed_large"),
        (None, "zero_qk"),
    ],
)
def test_state_flash_attention_interleaved_qkv(scale, pattern):
    generator = torch.Generator().manual_seed(20261105)
    storage = torch.randn(
        (8, 2049, 3, 16, 128), dtype=torch.bfloat16, generator=generator
    )
    if pattern == "mixed_large":
        # Exercise stable fallback and bounded tiles in the same invocation.
        storage[0].mul_(8)
    elif pattern == "zero_qk":
        storage[:, :, 0].zero_()
        storage[:, :, 1].zero_()
    device_storage = storage.to(flag_gems.device)
    shape = (8, 16, 2049, 128)
    stride = (12589056, 128, 6144, 1)
    cpu_inputs = [
        storage.as_strided(shape, stride, offset) for offset in (0, 2048, 4096)
    ]
    inputs = [
        device_storage.as_strided(shape, stride, offset) for offset in (0, 2048, 4096)
    ]
    if flag_gems.device == "flagos":
        # This Torch-FL build skips codegen for the exact flash ATen entry;
        # explicitly bind its PrivateUse1 entry for this test's lifetime.
        lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
        lib.impl(
            "_scaled_dot_product_flash_attention",
            flag_gems._scaled_dot_product_flash_attention,
        )
        result = torch.ops.aten._scaled_dot_product_flash_attention.default(
            *inputs, scale=scale
        )
    else:
        with flag_gems.use_gems(include=["_scaled_dot_product_flash_attention"]):
            result = torch.ops.aten._scaled_dot_product_flash_attention.default(
                *inputs, scale=scale
            )

    # Every batch and head, interior tile boundaries and the final odd row.
    rows = torch.tensor([0, 63, 64, 127, 128, 1024, 2047, 2048])
    q, k, v = [x.float() for x in cpu_inputs]
    actual_scale = 128**-0.5 if scale is None else scale
    scores = q[:, :, rows] @ k.transpose(-1, -2) * actual_scale
    reference = torch.softmax(scores, dim=-1) @ v
    reference_lse = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(
        result[0].cpu().float()[:, :, rows], reference, atol=0.002, rtol=0.02
    )
    torch.testing.assert_close(
        result[1].cpu()[:, :, rows], reference_lse, atol=2e-5, rtol=2e-5
    )
    assert len(result) == 9
    assert result[2] is None and result[3] is None
    assert result[4:6] == (2049, 2049)
    assert result[0].shape == shape and result[0].dtype == torch.bfloat16
    assert result[1].shape == shape[:3] and result[1].dtype == torch.float32
    assert result[6].shape == (2,) and result[6].dtype == torch.int64
    assert result[7].ndim == 0 and result[7].dtype == torch.int64
    assert result[8] is None
