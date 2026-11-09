# aten::mm: review blocks acceptance

PR #2 is a draft. The previous claim that all target cases were accepted is
withdrawn. Historical traces and test counts do not establish current correctness
or performance acceptance. The review feedback below is supplied by the user;
it is not presented as a new measurement by this checkout.

| Input | Review result | Acceptance |
|---|---|---|
| FP32 [60025,128] x [128,128], transposed B | Old FG, new FG and DAS each fail 1/7,683,200 elements | Correctness unresolved; performance not accepted |
| FP64 [1344,256] x [256,128], B offset 131072 | New FG fails 172,032/172,032 outputs; old FG and DAS pass; old 82.08 us | Severe new correctness regression; blocks merge |
| Same FP64 logical inputs, B offset 0 | New FG still fails every output; old 81.92 us | Offset is not the sole cause; blocks merge |
| FP64 [192,128] x [128,128] | 44.64 -> 8.16 us; DAS/new 105.88% | Review reports accuracy and stability pass |
| FP64 [282624,15] x [15,3] | 754.80 -> 44.08 us; DAS/new 103.81% | Review reports accuracy and stability pass |
| FP32 [28852,1] x [1,32] | 14.56 -> 4.00 us; DAS/new 174.00% | Review reports accuracy and stability pass |
| FP32 [6432,3] x [3,9] | 8.32 -> 2.72 us; DAS/new 147.06% | Review reports accuracy and stability pass |
| BF16 [32,64] x [64,32], fallback control | Accuracy passes; A/A max drift 25.64%; 11.36 -> 10.88 us diagnostic | Performance not accepted |

The FP64 failure is associated with the `mm_mmac_peel` path and the current
compiler stack. A source-only or compiler-only root cause has not been proved.
The reported failure was reproduced by the reviewer on two nodes. Passing a
different input or environment does not resolve that report. Do not relax
accuracy tolerances to convert a failure into acceptance.

The current correction replaces all three selected `peel` configurations with
the existing explicit-layout `swizzle` kernel, pipelined loads, prefetch 8 and
two warps. It bypasses the reported compiler-sensitive path without asserting
a source-only or compiler-only root cause. The original failure has not been
reproduced in this isolated environment, and is NOT marked resolved. The exact
reviewer input, test entry point and environment still need to be matched.

The second correction tunes only spreadsheet rows 20 and 32. Row 20 uses a
swapped explicit-layout swizzle launch with four warps; row 32 uses the existing
pipelined two-warp swizzle configuration. Both shapes are added to the
multi-seed, offset and low-mantissa-bit regressions. On a01r3n13 card 0 this
source passes **33 focused tests and 30 public-route checks**. The 19 targeted
spreadsheet cases pass both generated seeds (38 accuracy records). With a new,
separately recorded 200-call/ROI protocol, rows 20 and 32 reach **91.49% and
103.52%** DAS/new, with both sides below the 5% A/A guard. Overall, **11/19**
targeted groups pass this protocol; rows 9, 81, 82, 101, 107, 108, 111 and 114
remain unstable. These results do not establish acceptance of all 94 targets.
The separate BF16 fallback control also fails DAS stability and is diagnostic.

See [second-correction status](correction-20261109/phase2-long/status.json),
[all targeted samples](correction-20261109/phase2-long/device-samples.json),
[source identity](correction-20261109/phase2-long/source-proof.json) and
[original spreadsheet mapping](correction-20261109/phase2-long/spreadsheet-scope-audit.json).
The original FP32 and FP64 review failures remain unresolved. A separate
a01r4n15 run uses physical cards 0 and 1 in independent processes; every
operator itself runs on one card. The completed results are recorded below.
Formatting, full pre-commit and the required MM static checks pass.

On a01r4n15, the independent PR #2 source passes 33 focused tests and 30 public
route checks. Full-scope generated accuracy covers all 120 FP32/FP64 cases and
two seeds: **240/240 records pass** for DAS and new FG under the unchanged
recorded tolerances. The full 200-call/ROI performance run passes both A/A and
the 90% target for **44/94** groups; 50 fail stability. Those failures remain
in the evidence. A separately recorded 1024-call/ROI protocol passes all 22
preselected focus cases, and then all 36 uncovered unstable cases split across
the two cards. In total, **94/94 target groups have a passing device-performance
record**, using 36 groups from the 200-call protocol and 58 from the 1024-call
protocol. This is coverage across different burst lengths and cards, not one
uniform full-scope run. Source hashes match the same production implementation.
Generated inputs still do not close the original reviewer failures, and the PR
remains a draft.

See [complete cross-node classification](correction-20261109/crossnode-r4n15/status.json),
[retained full 200-call results](correction-20261109/crossnode-r4n15/device-samples-full-200.json),
[focused 1024-call results](correction-20261109/crossnode-r4n15/device-samples-focused-1024.json),
[card-0 remaining results](correction-20261109/crossnode-r4n15/device-samples-remaining-1024-card0.json),
[card-1 remaining results](correction-20261109/crossnode-r4n15/device-samples-remaining-1024-card1.json)
and [all measured source proofs](correction-20261109/crossnode-r4n15/source-proof.json).

The first-correction source on a01r3n13 visible card 0 passed 25 focused tests.
Eighteen public `torch.mm` checks cover three seeds and B offsets 0/131072 for
each former peel case. Every call witnesses one FG export and one explicit-layout
kernel launch, and checks every output against CPU FP64 at atol=rtol=1e-10.
The original 42-check diagnostic did not reproduce the old peel failure either;
its passing result is not evidence that the reviewer report was incorrect.

The expanded scope run checks all 120 FP32/FP64 spreadsheet cases with seeds
100003 and 880301. All 240 case/seed records pass for DAS and the new FG export
under the recorded tolerances (FP64 atol=rtol=1e-10; FP32 atol=2e-4, rtol=1e-4).
These generated inputs do not resolve the reviewer's shared FP32 1-element failure.

In that first-correction run, only **75/94** target groups passed both the 90% paired DAS threshold and the
newly declared 5% A/A guard in the single-compute-unit gate protocol. Rows 20
and 32 remain below 90%; 17 further rows fail stability for FG and/or DAS. They
are diagnostic, not accepted speedups. The three replacement configurations
separately measured DAS/new at 96.00%, 102.18% and 107.17%, with FG A/A drift
below 0.3%, in the confirmation run. Conditions differ from the user review
table; these measurements must not be combined with it.

See [current status](correction-20261109/revalidation-status.json),
[device samples](correction-20261109/device-samples-single-cu.json),
[public route checks](correction-20261109/public-route-checks.json) and
[source proof](correction-20261109/source-proof.json). The scope performance run
applies both PR source patches; the focused tests and public-route checks also
run against PR #2 independently. The PR remains a draft while the original
failure report and outstanding performance/stability cases remain unresolved.

For all new acceptance, require full value/layout checks before performance,
actual kernel/route evidence, matching valid DAS semantics, same-device paired
measurements and independent A/A blocks. Sum all constituent device kernel
durations and exclude host and gate time. Unstable timings remain diagnostic.
The identity/no-kernel case must not be assigned a speedup ratio.

Original reports are preserved in `historical/`. Existing raw result and
validation documents are historical snapshots, explicitly marked as superseded.
Source/hash/AST checks prove source identity only, not runtime correctness.
