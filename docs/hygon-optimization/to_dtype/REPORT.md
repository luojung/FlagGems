# aten::to.dtype: performance acceptance under review

PR #3 is a draft. The previous blanket acceptance and claim that the new cast
is uniformly faster are withdrawn. The following is user-supplied review data,
not a new measurement by this checkout. All times are device kernel times.

| Excel row / conversion | Review result | Acceptance |
|---|---|---|
| 1711 BF16 -> FP32 [1,145,2560] | 3.84 -> 3.04 us diagnostic; A/A max drift 6.25% | Performance not accepted |
| 1712 BF16 -> FP32 [1,4,145,256], stride [148480,256,1024,1] | New FG preserves layout; old public FG and DAS do not; new 2.56 us diagnostic; A/A fails | Layout fix reported; performance not accepted |
| 1713 BF16 -> FP32 [1,8,145,256], stride [296960,256,2048,1] | New FG preserves layout; old public FG and DAS do not; new 2.88 us diagnostic; A/A fails | Layout fix reported; performance not accepted |
| 1714 BF16 -> FP32 [2560] | 3.04 -> 2.40 us diagnostic; A/A max drift 6.46% | Performance not accepted |
| 1715 BF16 -> FP32 [256] | 2.5595 -> 2.32 us; 1.10x; DAS/new 110.34%; generic path | Review reports accuracy and stability pass |
| 1716 BF16 -> BF16 scalar [] | Returns input; no device kernel | Identity; speedup and DAS ratio undefined |
| 1717 BOOL -> INT32 [1,145] | 7.76 -> 3.04 us diagnostic; A/A max drift 7.23% | Performance not accepted |
| 1718 FP32 -> BF16 [1,145,256] | 2.64 -> 3.52 us diagnostic; new cast_dense; A/A max drift 6.18% | Performance not accepted |
| 1719 FP32 -> BF16 [1,8,145,145] | 3.84 -> 4.00 us; old/new 0.96x; DAS/new 92.00% | Accuracy/stability pass; DAS target met, but 4.17% slower than old public FG; 1.80x vs old generic |
| 1720 INT64 -> FP32 [1,145,1] | 7.60 -> 3.04 us; 2.50x; DAS/new 194.74%; old generic also 3.04 us | Accuracy/stability pass; benefit mainly from routing |

Keep old public FG, old generic direct, routing-only, new specialized cast,
and DAS measurements separate. A routing gain is not a new-kernel gain.
For the two permuted inputs, a public baseline that changes output strides
is not semantically equivalent for layout acceptance. A low-level
`_to_copy.out` boxing reference with explicit preserved strides is an additional
reference and must not be relabeled as the same public baseline.

Fresh measurements on a01r3n13 are in progress. Each provider requires full
value/dtype/stride/alias checks, actual route/kernel evidence, independent A/A
blocks and device-only timings with all constituent kernels included. Failed
stability checks must not produce an accepted speedup. The new measurement
protocol must be reported separately from the user's review conditions.

Original reports are preserved in `historical/`. Existing raw result and
validation documents are historical snapshots, explicitly marked as superseded.
Source/hash/AST checks prove source identity only, not runtime correctness.
