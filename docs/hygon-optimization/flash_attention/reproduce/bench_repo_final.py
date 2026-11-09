import hashlib
import json
from pathlib import Path

import flash_attn_2_cuda as fa
from probe import Timer, accuracy, flag_gems, make_inputs, torch, torch_fl

from flag_gems.runtime.backend._hygon.attention_state_hip import _load_state_kernel

inputs, cpu = make_inputs()
views = [torch_fl._C._flagos_to_cuda_view(x) for x in inputs]
wv = [x.transpose(1, 2) for x in views]
wheel = lambda: fa.fwd(
    *wv, None, None, 0.0, 128**-0.5, False, -1, -1, 0.0, False, None, False, None, 0.0
)
native = lambda: torch.ops.aten._scaled_dot_product_flash_attention.default(
    *views, scale=128**-0.5
)
repo = lambda: flag_gems._scaled_dot_product_flash_attention(*inputs, scale=128**-0.5)
library = _load_state_kernel()
assert library is not None
binding = torch.library.Library("aten", "IMPL", "PrivateUse1")
binding.impl(
    "_scaled_dot_product_flash_attention", flag_gems._scaled_dot_product_flash_attention
)
repo = lambda: torch.ops.aten._scaled_dot_product_flash_attention.default(
    *inputs, scale=128**-0.5
)
y = repo()
torch.flagos.synchronize()
check = accuracy(*y[:2], cpu)
root = Path(flag_gems.__file__).resolve().parents[2]
files = [
    "src/flag_gems/runtime/backend/_hygon/attention_state_hip.py",
    "src/flag_gems/runtime/backend/_hygon/attention_state_hip.cpp",
    "src/flag_gems/runtime/backend/_hygon/ops/attention_state.py",
    "src/flag_gems/runtime/backend/_hygon/ops/_scaled_dot_product_flash_attention.py",
    "tests/test_flash_attention_hygon_state.py",
]
hashes = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}
result = {
    "repo_accuracy": check,
    "source_sha256": hashes,
    "binary": library._name,
    "binary_sha256": hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),
    "timings": {},
}

timer = Timer()
for label, fn in [
    ("wheel", wheel),
    ("aten", native),
    ("repo", repo),
    ("wheel_repeat", wheel),
    ("aten_repeat", native),
    ("repo_repeat", repo),
]:
    result["timings"][label] = timer.bench(fn, label, rounds=5)
    print("TIMING", label, result["timings"][label], flush=True)
Path("repo_final_profile_validation.json").write_text(json.dumps(result, indent=2))
