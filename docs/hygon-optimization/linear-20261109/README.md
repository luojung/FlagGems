# Linear: fixed-source Hygon forward measurements (2026-11-09)

This change preserves the earlier metadata-guarded BF16 linear specializations and adds two measured optimizations. It does not declare full-domain accuracy/performance acceptance.

- For FP32 M=K=1 addmm with a vector bias and unit alpha/beta, use a strided vector affine kernel. The old matrix kernel already fused bias; the change avoids matrix tiling/dot setup and LDS. The selected launch uses BLOCK256, four warps and disabled FP fusion.
- For BF16 (M,K,N)=(145,2560,262208), change the existing guarded tile from (128,128,64,2,4) to (256,256,32,2,8). Other configurations and kernel bodies remain unchanged in this increment.

## Results

All 109 captured metadata cases were replayed as eager forward calls through public aten::linear.default: 54 model cases and 55 size-expansion cases. Baseline/candidate/DAS each collected 50 unprofiled API samples and ten strictly linked full-call kernel ROIs per case, on the same Hygon gfx936 physical device. Values are arithmetic means of all samples. Inputs were reconstructed, not original activations; captured requires_grad and forward tape were checked, but backward and whole-model behavior were not validated.

| Case | Baseline kernel us | Candidate kernel us | Kernel speedup | API speedup |
|---|---:|---:|---:|---:|
| model54-r7 | 7.440 | 2.992 | 2.487x | 1.307x |
| expanded55-r351 | 8.400 | 3.840 | 2.188x | 1.332x |
| expanded55-r352 | 11.312 | 3.536 | 3.199x | 1.344x |
| expanded55-r378 | 32.432 | 6.512 | 4.980x | 1.361x |
| expanded55-r399 | 407.968 | 64.480 | 6.327x | 2.098x |
| model54-r55 | 2968.799 | 2536.512 | 1.170x | 1.138x |

A subsequent fixed-source ABBA author recheck retained the gain direction for these six cases. It is not independent confirmation or formal AI4Perf promotion.

## Accuracy and limits

91/109 cases pass the declared default-tolerance reference; the same 18 FP32 cases fail on baseline, candidate and DAS, matching the initial baseline failure set. No new failure appeared in this replay. The 18 stay disqualified; all 109 retain measurements, with failed-row timings diagnostic only. BF16 final reference uses exact upcast CPU FP32 accumulation and one BF16 rounding (predeclared V3), while FP32 keeps CPU FP32. The earlier native-BF16 reference results remain separate and passed all nine BF16 baseline cases.

The current point estimates meet T_DAS_kernel / T_FG_kernel >= 0.9 in 15/54 model cases and 22/55 expanded cases, conditioned on numerical validity. R55 is still only 79.09% of DAS. An unchanged r51 path shows an unresolved ~5.3% API slowdown in the ABBA aggregate; process-level variation prevents a causal no-regression claim.

Eight additional finite mantissa/cancellation controls with non-unit strides and storage offsets passed on Hygon, including input preservation, output non-aliasing and public forward tape. The new pytest cases preserve the numerical/layout part as a repository regression test; that pytest wrapper was statically checked locally, not rerun on hardware. Production-file Black/isort/flake8 and four repository checks passed. Full-repo pre-commit hit its 180-second cap, so it is not an all-CI pass. NVIDIA and other backends are not hardware-validated by this report.

## Provenance

The incremental baseline mm SHA256 is 409211f396f29e878a28689304dba88cfa0d6a5ea340962c39d7226083d8f3b4. The tested final mm/addmm hashes are b10b0694aa527402358cfcf01ce8bddc7b26bd642fef907602adda975972648b and e50835830781eb4ab877611914c17db0b26eaa056deec48770a57b9f780a6975. Formatting-only delivery versions have identical full ASTs and parse as Python3.10. Private raw evidence is retained outside this repository; no private tensors, scheduler scripts or shared-environment configuration are published.

The first commit preserves the pre-existing BF16 implementation. The next commit contains the K1 branch, R55 tile change, regression controls and this report, keeping prior work separate from new gains.
