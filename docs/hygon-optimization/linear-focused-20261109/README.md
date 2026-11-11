# Hygon linear: FP32 projection specialization (2026-11-09 continuation)

Large FP32 projections kept dimensions and strides dynamic in the Hygon mm kernel. This increment adds a narrowly guarded constexpr kernel with the same IEEE arithmetic body, 64x64x32 tile, one stage, four warps and original materialization. The guarded (M,K,N) are (60000,512,1024) and (140000,512,512), with row-major A, column-major B and zero offsets after existing materialization. Other metadata keep their fallback. BF16 and K1 implementations are retained.

The baseline is the previous retained optimized source; earlier K1/BF16 gains are not credited again.

| Case | Previous kernel us | Retained kernel us | DAS kernel us | Latency reduction | DAS/FG | Correctness |
|---|---:|---:|---:|---:|---:|---|
| model54-r16 | 2589.903 | 1743.072 | 1455.376 | 32.70% | 83.49% | PASS |
| model54-r31 | 2784.911 | 1800.512 | 1463.232 | 35.35% | 81.27% | PASS |
| model54-r49 | 3258.703 | 2274.303 | 1932.943 | 30.21% | 84.99% | PASS |

All 109 metadata cases (54 model + 55 expanded) have fresh baseline/candidate/DAS API50 and strictly linked full-call kernel ROI10 measurements on the same Hygon gfx936 device. Performance and accuracy are independent: 43/109 reach DAS/FG>=0.90; 91/109 pass correctness; 2 performance-passing cases still fail correctness. All 18 FP32 failures match baseline and DAS, with no new failing IDs. The three specialized cases remain below 90%; aggregate threshold crossings in unchanged microsecond paths are not credited as new optimization wins. BF16 uses predeclared CPU FP32 accumulation and one BF16 rounding; tolerances are unchanged.

A frozen fresh-process author ABBA recheck covers four primary targets and five K1 cases. It is not independent confirmation. Eight bounded K1 controls and six BF16 controls pass. Reconstructed values are not original activations; forward/grad-enabled only. Backward, whole-model throughput and other hardware are not validated.

Returned FP32 code remains IEEE F32 MMAC: 79 registers/16KiB shared/zero spills versus 75 registers/16KiB in the dynamic baseline. The gain accompanies metadata specialization, not register reduction. Copies for r16/r49 remain included. The r34 generic addmm and r55 BF16 paths are unchanged; alternatives without benefit were excluded.

The exact measured source is [tested-sources/mm.py.txt](tested-sources/mm.py.txt). [SOURCE-PROVENANCE.json](SOURCE-PROVENANCE.json) binds measured bytes and formatting-only production source; full ASTs match and parse as Python3.10. Black/isort/flake8, four repository CI helpers and inventory unittest pass. Private raw statistics remain outside the repository; tensors and shared-environment configuration are not published.
