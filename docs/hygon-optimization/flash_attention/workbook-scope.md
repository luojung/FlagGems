# Worksheet scope and acceptance

The requested operator is `aten::_scaled_dot_product_flash_attention` only.
The workbook is `待优化设备耗时的算子1.xlsx`, SHA256
`d2596052c66d7145af47b35fda45198d1f213b69825aeb91d10112dfd8871597`.
Scanning both requested sheets finds four rows:

| Sheet | Excel row | Case | Q / K / V shape | Dtype | Stride |
| --- | ---: | --- | --- | --- | --- |
| 大于5%的模型工况 | 645 | model_state | `[8,16,2049,128]` | BF16 | `[12589056,128,6144,1]` |
| 规模扩展工况 | 2 | expanded_small | `[2,16,3,128]` | BF16 | `[18432,128,6144,1]` |
| 规模扩展工况 | 3 | expanded_medium | `[2,16,23,128]` | BF16 | `[141312,128,6144,1]` |
| 规模扩展工况 | 4 | expanded_large | `[2,16,362,128]` | BF16 | `[2224128,128,6144,1]` |

Every row has Q/K/V storage offsets `0/2048/4096`, scale
`0.08838834764831843`, no dropout and no causal masking. Full metadata and
expansion seeds are retained in `reproduce/workbook_cases.json`.

The workbook's worksheet dimension metadata incorrectly reports `A1`.
Read-only openpyxl readers must call `reset_dimensions()` before scanning;
the actual requested sheets have 898 and 953 rows respectively.

## Input provenance and limits

The model fixture uses a shared QKV backing based on the user's previous
review description. The workbook model note says the original alias/backing
contract and original operand payload are unavailable. Its seed is an
explicit reconstruction seed, not a recovered original-model seed.

The three expansion fixtures use independent minimum-size backing stores.
Their total storage sizes match the workbook exactly: 208896, 1683456,
and 26677248 bytes. They retain the worksheet seeds. CPU direct BF16 and
FP32-to-BF16 input generation are tested separately.

These are deterministic metadata reconstructions. They do not establish
that the missing original model payload or external frozen accuracy gate
has been replayed. Every new result retains these limitations explicitly.

## Common numerical and device performance contract

All output and LSE elements must pass the same reference comparison for
FlagGems, the native wheel entry, and native ATen. Output tolerances are
`atol=0.002, rtol=0.02`; LSE tolerances are `atol=rtol=2e-5`.
Nonfinite results, missing elements and shape mismatches fail acceptance.
An unqualified native result cannot supply the performance denominator.

The requested baseline is the DEV image default,
`2.8.3+das185.dtk2604.torch2100.2608132155.g9febea`. Its LSE comparison
has unresolved precision failures, so requested acceptance is incomplete.
The agent's substitution of SourceFind DAS nightly
`2.8.4+dtk2604.torch2100.2609292239.g74de14` was not user-authorized.
Archived version/hash-pinned 2.8.4 results are alternative experiments only.
Their eligible denominator cannot replace the requested DAS185 target.

Each card is evaluated independently on a01r4n15, using physical card 3 or
4. Final performance sampling uses 100 warmups followed by five rounds of
1000 calls per arm. Device time is the sum of hipprof HIPOPS kernel
durations, including helpers and recomputation; host overhead is excluded.
Both the mean of per-round native/FlagGems ratios and the ratio of mean
times must reach 0.9, using the faster qualified native entry.

PR #5 remains a draft. Alternative-baseline experiments and historical
single-fixture PR results do not establish completion. Requested acceptance
must use DAS185 with the unchanged full precision contract; no historical
native timing or alternate version may substitute for it. Current status
is in `acceptance-status.json`; evidence and limitations are in `REPORT.md`.
