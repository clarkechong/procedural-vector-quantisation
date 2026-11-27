# Experiment 02: scale, grammar length, and codebook geometry

Completed 2026-11-27 on one MI300X. This follow-up uses the same 4-weight tiles, 8-bit tile payload, checkpoint, 4,096 calibration tiles, and 4,096 original held-out tiles as Experiment 01. A further 4,096 final tiles were drawn from rows that occur in neither earlier split, across the same 21 matrices. [final_manifest.json](final_manifest.json) records the fresh sample. Program selection and scale fitting use calibration tiles only.

## Results

| Method | Original held-out NMSE | Fresh final NMSE | Effective bpw on 4,096 tiles |
| --- | ---: | ---: | ---: |
| Experiment 01 fixed-scale 2-op winner | 0.27463 | — | 2.00488 |
| Experiment 02 fitted-scale 2-op winner | 0.21841 | 0.20117 | 2.00488 |
| Guided 3-op signed-byte winner | 0.19544 | 0.18083 | 2.00830 |
| Guided 4-op signed-byte winner | 0.19502 | 0.18062 | 2.00977 |
| Guided 5-op signed-byte winner | **0.19483** | **0.18058** | 2.01123 |
| Guided 6-op signed-byte winner | 0.19583 | 0.18183 | 2.01465 |
| Four-level scalar as a 4-op program | 0.20989 | 0.19671 | 2.00977 |
| BF16-mask guided 2-op winner | 0.82337 | 0.83203 | 2.00488 |

NMSE is total squared reconstruction error divided by total original-weight squared energy. Each result uses exact enumeration of all 256 seeds per tile; FP32 GPU dot products select the seed for signed-byte codebooks, with FP64 residual accumulation. BF16 direct-output scoring uses FP64 throughout to avoid overflow from finite BF16 codewords. The original held-out and fresh final distributions differ slightly, so compare methods within a column.

Rates include the eight-bit tile payload, three bytes per instruction, a four-byte global scale for signed-byte methods, four FP16 levels for scalar, and four bytes per *new constant actually used* in the program. The guided 3–5-op winners use `0x8`, which adds four dictionary bytes relative to Experiment 01. The BF16 winner uses `0x3c00`, which also adds four bytes; it has no scale. These are experimental description-cost estimates; decoder implementation, shape metadata, and the base dictionary are excluded as in Experiment 01.

The fitted 2-op winner is `r0=ADD(i,i); r1=SHL(s,r0)` with global scale 0.00340023. The guided 3-op winner is `r0=SHR(s,i); r1=MUL(r0,0xf); r2=MUL(r1,0x8)` with scale 0.00380276. The 5-op winner adds `r3=ADD(i,r2); r4=ADD(i,r3)` and uses scale 0.00380344. Thus the 3-op program gives most of the quality gain at the lowest program cost. Its held-out score is slightly below the Experiment 01 Gaussian random VQ result of 0.19696, although that VQ costs 3 bpw over this sample and a single random-codebook draw is not a robust threshold.

On the fresh final set, the 5-op program improves NMSE over the scalar program by 0.01614. A paired tile bootstrap with 2,000 replicates gives a 95% interval of [0.01246, 0.01969] for the scalar-minus-5-op difference. The bootstrap treats sampled tiles as independent and does not capture variation across different checkpoints or layer selections. [analysis.json](analysis.json) records the calculation.

## Search coverage and controls

- **02A:** Every one of the 5,901 distinct signed-byte codebooks from Experiment 01 was fitted with three starting scales and five alternating seed-assignment/least-squares updates. This is local joint optimization, not a certified global optimum over continuous scale. [optimized_ranked.csv](optimized_ranked.csv) contains all fitted scores and geometry features. The best 2-op codebook changed after scale fitting.
- **02B:** The direct-BF16 grammar gained integer and BF16 constants, including `0x8000`, `0x7f80`, `0x007f`, `0x3c00`, and `0x3d00`. The bounded search evaluated 733 one-op syntaxes and 9,312 two-op syntactic extensions from 48 retained prefixes; 3,008 valid distinct two-op codebooks were screened and 96 were fully reranked. The best held-out result, 0.82337, is worse than the earlier narrower-grammar exhaustive result of 0.76515. This does not rule out useful longer BF16 programs. See [bf16_guided.json](bf16_guided.json).
- **02C:** A 24-wide beam searched 3–6 instructions, using 1,024 calibration tiles to screen, then all 4,096 calibration tiles to rerank the best 96 at each length. It screened 3,931, 3,621, 2,206, and 2,838 distinct codebooks at lengths 3–6. The search is **not exhaustive**. It began from the best fitted 2-op programs plus a hand-inserted scalar-extraction prefix. See [guided.json](guided.json).
- **02D:** The ordinary scalar codebook is explicitly representable as `r0=SHL(i,1); r1=SHR(s,r0); r2=AND(r1,3); r3=LUT4(r2)` with four learned FP16 lookup values. The resulting 256 codewords are exactly the Cartesian product of four 4-level scalar choices and reproduce the scalar baseline score. The LUT4 output stage is implemented in [followup.py](../../synthesis/followup.py) as a separate float lookup primitive; the signed-byte search grammar does not discover this program automatically. Shared description cost is 12 instruction bytes plus eight table bytes.
- **02E:** The target covariance is nearly isotropic: calibration PCA eigenvalues range from 0.03887 to 0.04337, and held-out values from 0.03826 to 0.04575. The Experiment 01 winner's output RMS varies from 33.6 to 73.9 across coordinates, with 56–59% zeros in coordinates 1–3. The guided 3-op winner has equal RMS 74.0 and 3.125% zeros at each coordinate, and its codebook covariance condition number is 1.13 versus 5.05 for the Experiment 01 winner. Across all 5,901 short codebooks, Spearman association between coordinate RMS imbalance and fitted NMSE is +0.39; within the best decile it reverses to -0.35. Geometry is informative but no single metric is a reliable standalone ranking rule. [analysis.json](analysis.json) contains all measured associations and winner metrics.
- **02F:** A separate 24-prefix two-op beam scored 1,565 distinct codebooks, versus 5,901 in the exhaustive two-op corpus. It recovered the same minimum calibration NMSE, 0.210953, using the equivalent program `r0=SHL(s,i); r1=SHL(r0,i)`. The beam therefore matched the exhaustive optimum in this specific audit while evaluating 26.5% as many distinct two-op codebooks. See [beam_audit.json](beam_audit.json).

## Limits and reproduction

These are weight-reconstruction results, not model perplexity or downstream accuracy. The 3–6-op and BF16 searches are bounded by beam choices and screening subsets. Scale fitting may converge to a local minimum. The scalar lookup primitive was constructed as a positive control, not found by the guided search. The final sample was not used for automated program selection, but results from it have now been observed and should not be reused as a blind set for future design decisions.

With the checkpoint and ROCm device exposed, run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.synthesis.followup --device cuda
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.synthesis.guided --device cuda
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.synthesis.bf16_guided --device cuda
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.synthesis.analyze_followup
.venv/bin/python -m unittest discover -s tests
```

Source files are [followup.py](../../synthesis/followup.py), [guided.py](../../synthesis/guided.py), [bf16_guided.py](../../synthesis/bf16_guided.py), and [analyze_followup.py](../../synthesis/analyze_followup.py). The JSON and CSV files in this directory contain measurements and actual search coverage.
