# Hygon aten::to.dtype vector cast reproduction

This experiment measures nine cast configurations through the public
`torch.ops.aten.to.dtype` entry point. The profiling worker installs the selected
implementation inside its own FlagGems context and verifies that the call reaches
it. Production dispatcher integration and validation after integration are
pending.

## Run the 35 cases

On a Linux compute node with an idle gfx936 GPU, use the FlagOS development
image `20261103-hygon-optimization`, DTK 26.04, Git, and Python 3 on the host.
Run from the PR checkout. Put the baseline clone, cache, and output on writable
scratch storage. The runner checks that the baseline's `src` tree is
`7d6ebe9c65b06ab3917fdc76142920f042d668f1` and clean.

```bash
export DEV=/work1/share/scnethpc2601/flagos-dev/20261103-hygon-optimization
export DTK_ROOT=/public/software/compiler/dtk-26.04
scratch=$(mktemp -d "${TMPDIR:-/tmp}/dtype-repro.XXXXXXXX")
git clone --filter=blob:none --no-checkout https://github.com/luojung/FlagGems.git "$scratch/base"
git -C "$scratch/base" checkout --detach 2862d83d0d8cd7d599f2059131309e6db554744d
export FLAGOS_DEV_CACHE="$scratch/cache"
bash docs/hygon-optimization/to_dtype/reproduce-vector-cast/run.sh \
    "$scratch/base" "$scratch/results" 0
```

The last argument selects one physical GPU. Use a fresh output directory for
each run. The script builds the supplied HIP source, runs the model and scaling
cases serially, and writes `results.csv`, per-provider logs and JSON summaries,
runtime fingerprints, and `profiles/{model,scale}.db`. The trace databases can
be large. `cases.json` records the Excel row, seed, dtype, shape, stride, and
storage offset for each input and makes the workbook unnecessary for reruns.
For a short check, prefix the last command with `MODEL_ROWS=775 SCALE_ROWS=1080`.

## Correctness and timing

- CPU ATen on the same seeded input is the correctness reference for each
  provider independently. The checks cover values, NaN locations, non-NaN
  floating-point bits, dtype, strides, device, and input/output aliasing.
- DAS uses Torch-FL boxing `aten::_to_copy` for conversion. FlagGems uses public
  `aten::to.dtype` with the pinned implementation and, where selected, the
  experimental implementation. Same-dtype scalar calls return the input and
  have no device conversion measurement.
- The harness reconstructs the device-only protocol from the workbook
  records. It reads HIP device activities inside profiler ranges, excluding
  host overhead and synchronization latency. There is no artificial queue
  gate. Three groups rotate provider order.
- Model cases run 20 calls per range and 25 ranges per group. Device time is
  the sum of kernel and device copy durations per call. The group score
  divides the DAS median by the FlagGems median.
- Scaling cases warm up for 100 calls and measure five ranges of 1,000 calls
  per group. The group score averages five paired ratios of DAS to FlagGems
  kernel time. The CSV reports median device times separately.
- Compare the lowest of the three group scores with 90%. The CSV requires
  both providers to pass correctness for a performance verdict. It reports
  the DAS stride failures separately and preserves their timing for diagnosis.

## Results

See [the 35-case device run](results.csv) for both correctness outcomes, device
times in microseconds, and all three group percentages.

The gfx936 run on `f13r3n07`, GPU 0, passed all 35 FlagGems correctness
checks. All 25 comparable conversions cleared 90% in all three groups.
The lowest score was 97.59% (row 1067). DAS passed 27 of 35 correctness
checks. Its output strides differed from CPU ATen for rows 772, 773, and
1059 through 1064. Rows 776 and 1068 are same-dtype scalar calls.

| Excel row | Pinned FG (us) | Candidate (us) | DAS (us) | Lowest group DAS / FG |
|---:|---:|---:|---:|---:|
| 775 | 2.840 | 2.040 | 2.312 | 112.89% |
| 779 | 6.136 | 2.360 | 2.704 | 114.14% |
| 1065 | 2.686 | 2.035 | 2.083 | 102.32% |
| 1066 | 2.919 | 2.468 | 2.502 | 101.33% |
| 1072 | 2.750 | 2.102 | 2.145 | 101.96% |
| 1073 | 2.784 | 2.341 | 2.341 | 99.93% |
| 1075 | 4.970 | 2.275 | 2.604 | 108.76% |
| 1076 | 4.070 | 2.654 | 2.907 | 109.42% |
| 1080 | 55.854 | 39.535 | 50.829 | 128.49% |
