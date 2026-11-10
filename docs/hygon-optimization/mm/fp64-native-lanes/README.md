# Hygon FP64 MMAC native-lane correction

This follow-up fixes 12 MACE FP64 `aten::mm` regressions introduced by the device-native kernels in luojung/FlagGems#2. The baseline old FlagGems and native boxing pass these reconstructed geometries while the original PR kernels fail. This branch is based on PR2 head `61fe8673f2afe2d30818243babe795a07cf1d2a6`; the two repaired production files retain the same executable AST as the hardware-validated private correction from 2026-11-09; import and f-string whitespace is normalized for the repository hooks.

## Change

Keep the native FP64 MMAC arithmetic. Load operands explicitly with `lane % 16` selecting rows/columns and `lane // 16` selecting each four-element K group, including replicated warps. In the swizzle path, consume one operand bank before advancing K instead of keeping overlapping prefetch banks live. Keep existing dispatch, shapes, strides, offsets, output stores and numerical tolerances. This does not route the affected shapes to old FG or native boxing.

Controlled diagnostics showed that explicit loads alone fix the coopfast case; the swizzle cases require both explicit loads and removal of overlapping prefetch. The inspected row 68 codegen still contains FP64 MMAC; LDS falls from 12288 to 4096 bytes and both inspected variants have zero scratch. This establishes a compatible correction for the pinned HCU path, not a proven single LLVM-pass defect.

## Reproduce

Use a Hygon gfx936 environment with the repository's supported FlagTree/HCU compiler and matching Torch-FL integration. Activate that environment and install this branch. The 36 regression combinations are 12 geometries times seeds 0/20261109/20261110:

```bash
python -c "import torch_fl, pytest; raise SystemExit(pytest.main(['-q', 'tests/test_mm_hygon_fp64_regression.py', '--ref', 'cpu']))"
```

The test recreates full CPU storage before migrating it, preserving non-contiguous strides and nonzero offsets, and compares complete FP64 output against CPU `torch.mm` using unmodified dtype-default tolerances. It calls the exported FlagGems `mm`, while the archived 98-case audit separately witnessed public ATen through a process-private Torch-FL `mm=flaggems` route. Original inter-input aliases were not captured; independent storage is an explicit reconstruction assumption. These are not original model operands or whole-model training acceptance.

## Validation and performance

- All 32 frozen FP64 geometries across 3 seeds pass: 96 complete comparisons; max absolute error 5.115907697472721e-13 (`accuracy-validation.json`).
- All 12 originally regressing geometries pass in all six full-measurement processes (`repaired-cases.json`).
- 98-case audit:29,400 device samples (98x6x50) and 19,600 separate profiler-free API samples (98x4x50); zero remaining new correctness regressions. Historical/shared numerical failures retain failure labels and diagnostic performance.
- Correctness and performance acceptance are separate: among the 12 corrected cases, only 2 performance points meet native/repaired>=0.90, and 1 pass the full correctness/stability/performance gate. Do not describe this fix as all 12 reaching90%.
- All device entries are arithmetic sums of producer-linked kernel durations. A fixed20,000-iteration atomic queue gate is excluded; memory is separate. Each provider gets50 samples in 2x25 with a reversed second-block arm order and its own repeat; any per-block mean A/A drift>5% remains labelled unstable. No retries selected for passing outcomes.

Raw per-sample values, all provider correctness flags, kernel names and A/A drift for the 12 cases are included in `repaired-cases.json`; the compact table is `regression-results.csv`. The broader98-case audit gives59 performance points>=90% and 7 full passes; it is not a claim of universal performance qualification.

## 底层运行逻辑与原理分析

CPU full backing -> native-stride/offset views -> one device migration -> eager/no_grad public `aten::mm.default` -> PrivateUse1/Torch-FL private mm routing -> Hygon `mm.py` and configuration -> repaired swizzle/coopfast load -> FP64 `v_mmac_16x16x4_f64` -> existing output store -> full CPU comparison. No Dynamo/Inductor fusion is involved. Compilation, transfers, reference computation, warmup and the queue gate are outside the primary kernel-duration boundary. Original and repaired trace/source witnesses distinguish route selection from kernel arithmetic. The narrow responsibility boundary is the vendor-specific FP64 operand-loading/prefetch implementation under this compiler; dispatcher and reference tolerance are unchanged.

## Submission checks (2026-11-10)

All changed-file pre-commit hooks and the four repository MM metadata/export/marker/ATen checks pass. Full-repository hooks were attempted: 30 existing E226 f-string whitespace errors remain in unmodified PR2 files; each offending line was verified against the base commit. The corresponding whitespace in the two touched kernels is corrected without changing their executable AST.

The new test matrix's 36 geometry/seed combinations were checked against the completed 96 hardware comparisons: all are covered and passed there. A separate execution of the new pytest file was queued, but Slurm reported AssocGrpGRES with no start time. That redundant job was withdrawn before allocation (zero GPU time). The new pytest entrypoint was therefore **not separately executed**; this submission relies on the completed hardware results and does not claim a fresh 36-test pytest pass. The matrix audit is included in accuracy-validation.json.
