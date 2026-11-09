# Hygon STATE FlashAttention device performance

The captured `aten::_scaled_dot_product_flash_attention` input reaches **94.57% of native DAS device performance**, exceeding the 90% target. This conclusion applies to the exact spreadsheet case below.

| Implementation | Device kernel time |
| --- | ---: |
| Native DAS wheel | 1620.60 us |
| Native ATen / DAS | 1620.79 us |
| FlagGems implementation | 1713.74 us |
| Maximum FlagGems time at the 90% target | 1800.67 us |

Performance is DAS device time / FlagGems device time. `repo_final_kernel_summary.json` contains the per-range kernel counts, names and durations: two alternating rounds of five profiler ranges, with ten calls per range. Every constituent kernel is included (key norms, HIP attention, stable fallback check). HIPOPS execution durations exclude host launch gaps, allocation, transfers, event time and initial compilation. Event timing emitted by reproduction scripts is diagnostic only; acceptance uses the profiler kernel sums.

## Exact input and implementation

- Spreadsheet sheet `工况明细`, row 646, STATE model.
- BF16 Q/K/V: shape `[8, 16, 2049, 128]`, stride `[12589056, 128, 6144, 1]`, shared-storage offsets `0/2048/4096`.
- Default scale `128**-0.5`, no dropout, non-causal, no debug mask.
- Original measurement: node a01r3n18, gfx936, physical GPUs 4 and 5, DTK 26.04 development image.
- Native reference: `flash_attn==2.8.3+das185.dtk2604.torch2100.2608132155.g9febea`.

The backend specialization uses autonomous HIP MMAC QK/PV computation with FP32 accumulation and BF16 probability rounding. It prefetches K/V and reuses LDS for coalesced output stores. A device Cauchy bound controls fixed-shift softmax; unsafe tiles are recomputed with stable Triton softmax. No DAS compute is called by the optimized implementation, and no tensor contents are assumed.

The HIP path requires gfx936 and the default scale. Other scales and unavailable compiler configurations retain the Triton implementation; other input layouts retain the existing general attention path. The measured speed is not a claim about those fallbacks.

The optional HIP source is compiled on first use and cached under the user's `~/.cache/flag_gems/hygon_state_flash/`, with a build lock. The development image provides `/opt/flagos/toolchains/aillvm/bin/clang++`; `FLAG_GEMS_HYGON_CLANG` can override that path. Packaging explicitly includes the HIP `.cpp` source. No shared image changes are required.

## Correctness and integration

| Check | Recorded result |
| --- | --- |
| Full output vs CPU FP32 SDPA | 33,570,816 elements pass; max absolute error 0.00155908 |
| Full LSE vs CPU FP32 logsumexp | 262,272 elements pass; max absolute error 1.90735e-6 |
| Large-norm tiles and zero Q/K | Pass; stable fallback exercised |
| Focused GPU tests | 6 pass: default/explicit-default/positive/negative scale, mixed-large and zero-QK patterns |
| ATen return contract | Nine return values, shapes and dtypes checked |

Output tolerance is `atol=0.002, rtol=0.02`; LSE tolerance is `atol=2e-5, rtol=2e-5`. `repo_final_validation.json` and `pytest-current.log` preserve the results. The large-norm full-validation guard receipt uses sampled rows; the focused tests additionally check the explicit tolerance contract.

This Torch-FL image does not generate a PrivateUse1 kernel for the exact flash ATen entry. The test and profiler script explicitly register the FlagGems implementation for their process, then call `torch.ops.aten._scaled_dot_product_flash_attention.default`. The performance result therefore verifies that bound ATen implementation; automatic routing in an unmodified Torch-FL process remains an integration limitation. Native DAS uses Torch-FL's zero-copy CUDA view.

The hashes in `flash_state_provenance.json` identify the five originally measured implementation/test files. Repository formatting, import sorting and Git newline normalization change some PR file hashes; `source-equivalence.json` records both versions and verifies Python AST equivalence after import ordering, or identical C++ tokens. This PR also adds documentation and package data without changing the measured computation. Historical GPU checks were run in the authorized development environment; preparing this PR does not constitute a fresh GPU rerun.

## Reproduction

Use the same DTK/Torch-FL development image and the specified DAS wheel. Run on a compute node with an authorized resource allocation, selecting the available physical GPU before launching the image. The original environment runner accepts `--source /path/to/this/checkout`.

Run the focused test from the repository root:

```bash
pytest tests/test_flash_attention_hygon_state.py --ref cpu -q
```

`reproduce/probe.py`, `bench_repo_final.py` and `validate_repo_final.py` preserve the measurement inputs and validation procedure. Their shared helper expects the DTK runtime at `/public/software/compiler/dtk-26.04`; adapt that runtime location when using a different image. From a separate writable output directory, with the reproduction directory on `PYTHONPATH`, run:

```bash
python /path/to/reproduce/validate_repo_final.py --mode tune
hipprof --hip-trace --hiptx-trace --output-type 0 -o profile \
  python /path/to/reproduce/bench_repo_final.py --mode tune --profile
python /path/to/reproduce/analyze_profile.py profile.json
```

The analyzer checks the expected kernel counts and recomputes the device-only result. Keep the resulting `repo_final_profile_validation.json` beside the trace when analyzing it. Full traces remain in the original measurement workspace; the compact per-range results are included here.
