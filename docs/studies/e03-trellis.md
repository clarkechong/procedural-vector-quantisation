# Experiment 03: overlapping latent windows

The three-operation Experiment 02 winner is exactly `int8(120 * (seed >> index))` for all 256 eight-bit seeds and all four coordinates. Constant folding gives `SHR` followed by `MUL(120)`. All 256 codewords match. Each coordinate depends on five seed bits; adjacent coordinates share four bits. The exact influence matrix is in [results.json](../../results/e03-trellis/results.json).

The tested extension uses an open bitstream boundary. The first output consumes `L` bits and every later output consumes `k` new bits, so the payload for `N` weights is `L+k(N-1)` bits. The encoder uses an exact `2^L`-state Viterbi recurrence for a fixed map and scale. Independent exhaustive enumeration matched its error on small tiles, including the discovered four-weight geometry. A codebook equivalence test also matches the original three-operation generator. [Source](../../experiments/synthesis/trellis.py) · [Tests](../../tests/test_trellis.py)

## Fresh-row rate–distortion

Calibration and test rows were drawn from the `down_proj` matrices at layers 0, 15, and 29 of checkpoint SHA-256 `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`. They do not occur in either earlier experiment's manifests. Calibration and test use disjoint rows. The same pools recur at different lengths, but splits never share a row. Program and multiplier choices use calibration only; the test metrics were recorded once. These scores therefore describe this checkpoint's selected down projections, not the seven-projection mixture in Experiments 01–02. [Sample manifest](../../results/e03-trellis/sample_manifest.json)

| Weights `N` | `L=5,k=1` payload bpw | Discovered map test NMSE | `L=5,k=2` payload bpw | Same map, `k=2` test NMSE | Test rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2.0000 | 0.1682 | 2.7500 | 0.0955 | 64 |
| 8 | 1.5000 | 0.2347 | 2.3750 | 0.1034 | 64 |
| 16 | 1.2500 | 0.2661 | 2.1875 | 0.0991 | 64 |
| 32 | 1.1250 | 0.3065 | 2.0938 | 0.1192 | 48 |
| 64 | 1.0625 | 0.3417 | 2.0469 | 0.1265 | 32 |
| 128 | 1.0312 | 0.3504 | 2.0234 | 0.1544 | 24 |
| 256 | 1.0156 | 0.3161 | 2.0117 | 0.1112 | 16 |
| 1,024 | 1.0039 | 0.3271 | 2.0029 | 0.1223 | 8 |

The global scale is fitted separately for each length and map with three starts and two alternating Viterbi/least-squares passes. The Viterbi result is exact **given** that fitted scale, while scale fitting is local. The table counts payload bits only. A shared scale and map description add small finite-sample overhead. Byte-aligning every row, as in the model codec, increases the `N=1,024` stored rate from 1.0039 to 1.0078 bpw.

[Rate–distortion plot](../../results/e03-trellis/rate_distortion.png) · [SVG](../../results/e03-trellis/rate_distortion.svg) · [Plot source](../../experiments/synthesis/plot_trellis.py)

On the same 64 four-weight test rows, a locally fitted four-level scalar control scored 0.1995 NMSE versus 0.1682 for the discovered map, both at two payload bits per weight. The paired scalar-minus-window NMSE difference was 0.0313; a 2,000-replicate row bootstrap interval was [0.0131, 0.0495]. At 1,024 weights, the discovered `k=1` map scored 0.3271 versus 0.4039 for a two-level scalar control on the same eight test rows, at about one payload bit per weight. The four-level scalar control scored 0.1387 at two bits per weight. The smaller long-tile samples are descriptive, not strong cross-model evidence. All scalar levels were fitted on calibration rows. [Matched controls](../../results/e03-trellis/results.json)

## Mapping search and QTIP-style control

The 16 multipliers `8 × odd` were screened as computational five-bit maps on the new four-weight calibration rows. The original multiplier **120 ranked first**, at 0.1252 calibration NMSE; 104 ranked second at 0.1313. This is a narrow structured search, not a full alternative-program synthesis. The unrestricted generator search remains represented by the separate Experiment 02 beam results.

The repository's 1MAD map, used as a QTIP-style core control, was encoded with the same Viterbi routine and sampled rows at state widths 8, 12, and 16 and strides one and two. At `N=1,024`, the 16-bit, stride-two 1MAD core scored 0.0867 NMSE at 2.0137 payload bpw on four test rows. The discovered map with stride two scored 0.1390 on those same four rows at 2.0029 payload bpw, after a separate scale fit; its eight-row score was 0.1223. A fixed signed Hadamard transform of 64-weight groups changed the 1MAD core's 1,024-weight NMSE to 0.0804 on the same four rows, while worsening its scores at lengths 64 and 128. These small long-tile samples do not establish a robust ranking. The full table contains 84 width/stride/length measurements, including `L=4,5,8,12,16` and both strides. This is a local QTIP-style control, **not** a reproduction of QTIP's full calibration, transforms, or model pipeline.

## Program analysis and hardware

Constant folding reduced the guided 3-, 4-, 5-, and 6-operation winners by one multiply each; all resulting 256-entry codebooks were checked exactly. The exhaustive Boolean influence analyzer identifies which seed bits affect each coordinate in any supplied bitvector program up to 16 seed bits. A lookup-based analyzer gives the corresponding matrix for windowed maps, including bit cancellations. [Simplifier](../../experiments/synthesis/simplify.py) · [Analysis runner](../../experiments/synthesis/trellis_analysis.py)

A Triton kernel fused packed-window decoding with one dot product per row on the ROCm `gfx942` device. For 4,096 rows of 1,024 weights, it took 0.00669 ms per launch in an isolated microbenchmark, about 627 billion decoded weights/s. A PyTorch dense FP16 matrix-vector control on the same shape took 0.01457 ms and read an 8,388,608-byte weight matrix versus the window kernel's 528,384-byte payload. Sixteen rows agreed with the CPU decoder/dot product to maximum absolute difference 1.2e-6. This is a kernel microbenchmark, not model throughput or a dense GEMM comparison. [Benchmark result](../../results/e03-trellis/fused_benchmark.json) · [Kernel](../../experiments/hardware/window_decode.py)

## Model-level smoke tests

The discovered map was applied to all 210 SmolLM2-135M projection matrices at strides one and two. One FP32 scale was fitted from eight 64-weight spans per matrix, and every row was encoded by exact Viterbi at that scale with byte-aligned storage. Rates include the matrix scales and row padding, but exclude the shared generator description.

| Method | Projection bytes | Stored bpw | Full projection weight NMSE | Perplexity, 127 targets | CPU encode time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original model | 212,336,640 | 16.0000 | 0 | 22.46 | — |
| Window map, `k=1` | 13,427,400 | 1.01178 | 0.32941 | 13,844,862 | 576.6 s |
| Window map, `k=2` | 26,698,440 | 2.01178 | 0.12090 | 7,629,407 | 537.8 s |

Both runs used the same **128-token smoke slice** and scored 127 next-token targets. The two-bit stride materially improves weight NMSE but still gives catastrophic short-slice perplexity. This rules out treating the weight-only gains as useful model-quality gains in this current codec. Full WikiText-2 perplexity was not run; 127 targets are too few for a stable exact estimate, while both failures are clear on this slice. The codec uses a per-matrix scale and no activation calibration, rotation, or layer-specific mapping. [Stride-one metrics](../../results/e03-trellis/model-smoke/perplexity.csv) · [Stride-two metrics](../../results/e03-trellis/model-smoke-k2/perplexity.csv) · [Stride-one payload manifest](../../results/e03-trellis/model-smoke/window_120_payloads.csv) · [Stride-two payload manifest](../../results/e03-trellis/model-smoke-k2/window_120_payloads.csv) · [Model runner](../../experiments/evaluation/trellis_model.py)

## Reproduce

From the repository root, with the checkpoint and ROCm device available:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.synthesis.trellis_experiment
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.synthesis.trellis_analysis
MPLBACKEND=Agg MPLCONFIGDIR=/tmp .venv/bin/python -m experiments.synthesis.plot_trellis
.venv/bin/python -m experiments.evaluation.trellis_model --device cuda --dtype float32
.venv/bin/python -m experiments.evaluation.trellis_model --device cuda --dtype float32 --stride 2 --out results/e03-trellis/model-smoke-k2
.venv/bin/python -m experiments.hardware.window_decode
.venv/bin/python -m unittest discover -s tests
```
