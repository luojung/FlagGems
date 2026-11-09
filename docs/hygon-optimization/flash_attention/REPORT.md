# Hygon STATE FlashAttention: corrected acceptance status

**The previous 94.57% DAS performance claim is withdrawn. The DAS 90% target is not verified.** Native FlashAttention fails the frozen LSE accuracy gate and is ineligible as a performance denominator. Historical native timings must not be substituted.

## Current external review result

| Comparison | Device result | Acceptance |
| --- | --- | --- |
| Previous FlagGems to new FlagGems | 4664.80 us to 1722.00 us; **2.71x** | Review reports output/LSE correctness and stability pass for both |
| New FlagGems to native DAS | **No valid ratio** | Review reports **23.6% of native LSE elements fail** the frozen gate |

These numbers come from the external review summary supplied by the user. The raw review script/results have not yet been provided or independently rerun here. They supersede the author's previous claim of DAS acceptance. Benefit against the old FlagGems implementation is a separate result from DAS qualification. Only device kernel execution time is in scope; host overhead is excluded.

## Confirmed validation mistake

The author's earlier native checks used LSE `atol=1e-3, rtol=1e-4`, whereas the FlagGems tests used `atol=2e-5, rtol=2e-5`. The final timing script checked only the candidate against CPU before timing native. Native-vs-wheel equality is not a CPU correctness gate. Those checks did not establish that DAS meets the same numerical contract.

`acceptance_correction.json` records the current status. `repo_final_kernel_summary.json` preserves the earlier profile only under `historical_profile`, marked ineligible for DAS acceptance; its current accepted ratio and `meets_90` fields are null. The historical receipts remain available at PR commit `c8322225a8b33718361b42ee12b310011d8bb393` for audit, not reuse as a denominator.

## Implementation and input

The input is spreadsheet row 646, STATE BF16 Q/K/V `[8,16,2049,128]`, shared storage, stride `[12589056,128,6144,1]`, offsets `0/2048/4096`, default scale `128**-0.5`, no dropout, non-causal, no debug mask.

The implementation uses autonomous gfx936 HIP MMAC QK/PV with FP32 accumulation, K/V prefetch and coalesced output stores. A device norm bound selects fixed-shift softmax; unsafe tiles are recomputed with stable Triton softmax. Other input layouts use the existing general attention path; other scales or unavailable compiler configurations use Triton. No DAS computation is delegated by the optimized implementation.

The optional HIP source is included in package data, compiled with the development image compiler on first use and cached under the user's `~/.cache/flag_gems/hygon_state_flash/`. `FLAG_GEMS_HYGON_CLANG` overrides the default compiler path. The five measured sources and their formatted PR equivalents are recorded in `source-equivalence.json`; the correction changes validation and reporting, not the computational kernels.

Historical author checks cover all 33,570,816 output and 262,272 LSE elements for the candidate, plus six focused GPU tests. Those receipts establish only their recorded scope. The user-provided review additionally reports candidate and old-version correctness/stability. Neither provides an eligible native baseline.

The original performance profile used GPU 4; full candidate correctness validation ran on GPU 5. This is not a claim that each card independently passed the performance target. The original a01r3n18 allocation is now inaccessible; no fresh GPU run is claimed in this correction.

## Corrected reproduction safeguards

`reproduce/accuracy_gate.py` applies the original PR output tolerance (`atol=0.002, rtol=0.02`) and LSE tolerance (`atol=2e-5, rtol=2e-5`) identically to all three arms: wheel DAS, native ATen/DAS and new FlagGems. It checks every element against CPU FP32 SDPA output and CPU FP32 matmul/logsumexp LSE, records failed counts/percentages and nonfinite values, and rejects shape broadcasting or partial validation. These constants are preserved without relaxing native; agreement with the external review's exact frozen script is still pending.

`bench_repo_final.py` writes a fresh run/input/source/native-library identity before validation and does not emit any timing ranges unless all arms pass. Failed validation is written to the receipt; no historical timing fallback exists. `analyze_profile.py` requires the matching full gate and run ID, rejects old/mixed traces, and requires all five ranges in both rounds. While the external frozen gate is unconfirmed, it emits only a diagnostic ratio under the original common PR contract; the accepted DAS ratio and `meets_90` stay null.

CPU guard regressions run without GPU access:

```bash
python docs/hygon-optimization/flash_attention/reproduce/test_accuracy_gate.py
```

On an allocated authorized compute node, use the original DTK/Torch-FL image and installed DAS wheel, select a single GPU 4 or 5, and run the image with `--source /path/to/checkout`. Run in a new writable output directory with `PYTHONPATH` pointing to `reproduce/`:

```bash
hipprof --hip-trace --hiptx-trace --output-type 0 -o profile \
  python /path/to/reproduce/bench_repo_final.py --mode tune --profile
python /path/to/reproduce/analyze_profile.py profile.json
```

The shared runtime helper expects `/public/software/compiler/dtk-26.04`; adapt that location if necessary. A native gate failure intentionally stops acceptance timing. `validate_repo_final.py` and the focused GPU test remain candidate-only historical checks and cannot qualify DAS.

To complete the original target, a native DAS implementation/configuration must first pass the exact frozen gate. Then each tested card needs its own same-input, same-run device-only comparison. A faster previous FlagGems baseline or a historical native time cannot replace this evidence.
