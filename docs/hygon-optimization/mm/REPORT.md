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

On a01r3n13, an isolated checkout and fresh cache have so far passed 42 FP64
checks (three seeds, B offsets 0 and 131072, seven providers, full CPU FP64
reference, atol=rtol=1e-10). This has NOT reproduced or fixed the review failure.
The exact reviewer input, test entry point and environment still need to be
matched. The first new device-only comparison measured the proposed peel path
at 16.784 us against DAS 13.816 us (82.31%) for [1344,256] x [256,128], below
the 90% target. Alternative kernels are under investigation. These fresh
measurements use different conditions from the review table and do not replace
it. No kernel fix has yet been accepted.

For all new acceptance, require full value/layout checks before performance,
actual kernel/route evidence, matching valid DAS semantics, same-device paired
measurements and independent A/A blocks. Sum all constituent device kernel
durations and exclude host and gate time. Unstable timings remain diagnostic.
The identity/no-kernel case must not be assigned a speedup ratio.

Original reports are preserved in `historical/`. Existing raw result and
validation documents are historical snapshots, explicitly marked as superseded.
Source/hash/AST checks prove source identity only, not runtime correctness.
