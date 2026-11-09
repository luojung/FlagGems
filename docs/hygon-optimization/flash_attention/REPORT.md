# Hygon STATE FlashAttention: fresh single-card verification

The captured spreadsheet input passes the same full-output/LSE precision contract for native DAS and FlagGems on physical cards **3 and 4 of a01r4n15**, tested independently. Both exceed 90% of the pinned official DAS 2.8.4 baseline in fresh device-only profiles. This is a new, version-qualified comparison; the failed DAS 185 baseline and historical 94.57% claim remain withdrawn. The external review's raw script/input was not supplied and is not claimed to have been replayed.

## Current device results

All numbers below are fresh kernel execution durations in microseconds. Each card has its own gate, run ID and trace; no result combines the two cards. Card 3's complete run ended before card 4 began.

| Physical card | Native wheel | Native ATen | FlagGems, all helpers | min(native wheel, ATen) / FlagGems |
| --- | ---: | ---: | ---: | ---: |
| 3 | 1606.76 | 1607.05 | 1685.99 | 95.30% |
| 4 | 1606.83 | 1607.68 | 1684.41 | 95.39% |

The native baseline is the official SourceFind DAS nightly wheel **flash_attn==2.8.4+dtk2604.torch2100.2609292239.g74de14**, unpacked into an isolated user directory. Its extension and matching libflash_attention.so are both selected and hashed in every timing receipt. This is a **baseline version change**, not a repair or qualification of DAS 185. The shared image was not modified. A second official matching opt1 package also passed the common gate; the newer, faster 2.8.4 package is used for the final comparisons.

The previously installed **2.8.3+das185.dtk2604.torch2100.2608132155.g9febea remains ineligible**. The external review reports 23.6% LSE failures; our different fixed-seed fixture produces 40.1232% failures on both cards. These are different fixtures and are not claimed to reproduce the same percentage. Layout and official prefetch/HG-mode switches did not resolve that failure. Its historical 94.57% performance claim stays withdrawn; its timing is never reused here.

## Exact input and implementation fix

Spreadsheet row 646: BF16 Q/K/V [8,16,2049,128], one shared backing [8,2049,3,16,128], strides [12589056,128,6144,1], offsets 0/2048/4096, no dropout/causal/debug mask. The captured **scale is 0.08838834764831843**. The seed-20261105 backing SHA256 is `44ebfeb326f5a257e890668d149dd0e28a715052ab3d7560d14726f39107fc04`.

The old dispatch guard recognized only `128**-0.5` (0.08838834764831845). The captured `1/sqrt(128)` double differs by one ULP, causing the exact spreadsheet call to miss the HIP specialization even though both scales have the same FP32 representation. The guard now recognizes both exact expressions; other scales retain the Triton path. The captured literal is a new GPU regression case. The HIP MMAC kernel itself is unchanged. The norm helper now uses eight warps: a separate alternating device profile measured approximately 60 us versus 88-91 us with four warps, with all helper kernels included in the final independent-card profiles.

Torch-FL CUDA hooks and the native extension are explicitly initialized before FlagGems probes devices. This fixes the reproduction loader order. The current Torch-FL image still needs an explicit temporary PrivateUse1 binding for the exact ATen operator; these measurements exercise that bound ATen entry, and do not claim automatic routing works in an unmodified process.

## Precision, stability and timing

The common gate uses CPU FP32 SDPA output and FP32 matmul/logsumexp LSE. Every arm uses unchanged tolerances: output atol=0.002, rtol=0.02; LSE atol=rtol=2e-5. Every timing run checks **all 33,570,816 output and 262,272 LSE elements** for wheel DAS, native ATen and FlagGems before emitting timing ranges. All pass with zero mismatches and zero nonfinite values. The native and candidate LSE maximum errors on the timed fixture are approximately 1.91e-6.

Both cards additionally pass **7 GPU tests** covering implicit default, explicit default, exact captured scale, positive/negative scales, mixed-large fallback and zero Q/K. On each card, native DAS and FlagGems pass full output/LSE checks for seeds 20261105, 0 and 42, three repeats per seed, with bitwise repeatability. This is repeated for both direct BF16 generation and FP32 generation followed by BF16 rounding, covering six independent fixtures per card. Eight CPU regressions guard against relaxed, partial, stale and nonfinite acceptance evidence; they distinguish valid recorded-contract ratios from unverified external-review replay. Focused pre-commit and the four operator static checks pass. The earlier full-repository pre-commit attempt found an unrelated baseline import-grouping issue; it is not claimed to pass.

Device time is the sum of hipprof HIPOPS kernel durations inside five ranges of ten calls, repeated in two alternating rounds. The reported value is the median of each round's five range means, then the median of the two rounds. **All three FlagGems kernels** (norm, MMAC attention, stable fallback) are included. Host execution, allocation, transfers and compilation are excluded. Accuracy is checked before timing, and each trace must match its gate's fresh run ID. A failed gate stops the timing phase.

## Evidence and reproduction

`node15-verification/summary.json` indexes both cards' native identities, source hashes and results. Each card directory contains its full gate, kernel summary, raw compressed trace, kernel CSV, GPU test log and stability receipt. Wheel URL/version/size/SHA256 receipts are included. Run `python docs/hygon-optimization/flash_attention/reproduce/replay_bundle.py` from the checkout to verify both source identities and replay both compressed traces. The final measured source hashes match the committed source bytes, including line endings. Historical files outside this directory retain only their stated historical scope.

On the authorized node, unpack the pinned official wheel into a user directory, then run each card **sequentially**, using a new output directory each time:

```bash
bash docs/hygon-optimization/flash_attention/reproduce/run_profile.sh \
  /path/to/FlagGems /path/to/unpacked/native/package 3 /path/to/new-output-card3
# After card 3 finishes, repeat with card 4 and a different output directory.
cd /path/to/new-output-card3
python /path/to/FlagGems/docs/hygon-optimization/flash_attention/reproduce/analyze_profile.py profile.json
```

The isolated package selection in `run_profile.sh` must also surround `validate_stability.py`, `validate_fp32_fixture.py` and the focused pytest command. Run the stability scripts in separate output directories; both write `stability.json`. The image environment, physical-card masks and native package must match the recorded receipts.

The analyzer reports `meets_90` only for its recorded input, pinned native implementation and unchanged common precision contract. `external_frozen_gate_verified=false` remains explicit: the external review's raw input/script has not been replayed. A valid new comparison does not erase the earlier invalid-denominator mistake, qualify DAS 185, or establish performance on other layouts/scales.
