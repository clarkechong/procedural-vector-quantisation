# Experiment 01: exhaustive four-weight computational codebooks

Completed 2026-11-26 on one MI300X. Source: `experiments/synthesis/miniature.py`.

## Question and scope

Can a one- or two-instruction integer program turn an arbitrary eight-bit latent into a useful four-weight codebook? One program is shared across sampled SmolLM2-135M BF16 projection tiles. All 256 latent values are searched for every tile. Program selection uses 4,096 calibration tiles; a separate 4,096 tiles from disjoint matrix rows are held out.

This is an exact search over the *declared finite grammar*, not all conceivable one- or two-instruction programs. The operations are ADD, SUB, MUL, AND, OR, XOR, NOT, SHL, SHR, ROTL and POPCOUNT with uint32 arithmetic. Operands come from the eight-bit seed `s`, index `i` in 0–3, earlier registers and constants 0, 1, 2, 3, 15, 255. Enumeration uses `grammar.choices` and `grammar.extensions`, which omit identity and dead instructions and canonicalize commutative operands. The primary output reads the final low byte as signed int8 and multiplies it by one FP32 scale learned from calibration RMS/32. No per-tile exponent is reserved or fitted in synthesized programs. A second search reads the final low 16 bits directly as BF16 and rejects programs with NaN or infinity codewords.

The source checkpoint is `data/smollm2-135m.safetensors`, SHA256 `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`. Tiles were sampled from all seven projection types at layers 0, 15, 29. Calibration and held-out rows are disjoint within each of the 21 matrices. Sampling is recorded in [sample_manifest.json](sample_manifest.json).

## Results

| Method | Held-out NMSE | Effective bits/weight on 4,096 tiles | Shared bytes |
| --- | ---: | ---: | ---: |
| Best one-instruction program | 0.52079 | 2.00342 | 7 |
| Best two-instruction program | 0.27463 | 2.00488 | 10 |
| Best direct-BF16 two-instruction program | 0.76515 | 2.00293 | 6 |
| LFSR seed + exponent, 4 bits each | 0.34425 | 2.00195 | 4 |
| Standard hashed PRNG | 0.37639 | 2.00195 | 4 |
| Lloyd–Max, four FP16 scalar levels | 0.20989 | 2.00391 | 8 |
| Random Gaussian VQ, 256 FP16 codewords | 0.19696 | 3.00000 | 2,048 |
| Trained VQ, 256 FP16 codewords | 0.14778 | 3.00000 | 2,048 |

All methods use an eight-bit tile index, except the scalar method, whose four two-bit indices also total eight bits. Effective rate includes the listed shared bytes over 16,384 held-out weights. The VQ rows thus cost 3 bpw on this small sample; the extra codebook bytes would amortize over a larger deployment. The LFSR control is SeedLM-inspired, not the published SeedLM format. Scalar levels and VQ codewords are rounded to FP16 before held-out evaluation.

The best signed-byte one-instruction program is `r0=SHL(s, i)`. The best two-instruction program is `r0=MUL(s, i); r1=ROTL(s, r0)`. Their calibration NMSEs were 0.52883 and 0.28154, respectively. The two-instruction winner also had 0.25969 NMSE on independently shuffled held-out weights and 0.27632 on held-out IID Gaussian weights. Similar performance on these controls gives no evidence of a useful positional pattern in this sample. The direct-BF16 search had 5,187 distinct finite codebooks; its best program was `r0=OR(s, 0xf); r1=MUL(s, r0)` with 0.76515 held-out NMSE. [Direct BF16 rankings](direct_bf16_ranked.csv) record that search.

The two-instruction winner has 256 distinct codewords, affine rank four, effective rank 3.32, polynomial degree six in the eight Boolean seed bits, and mean absolute output change 20.36 signed-byte units when seed bit zero is flipped. These properties describe the complete codebook. [ranked.csv](ranked.csv) records every distinct codebook, its calibration scores and algebraic properties; [equivalent_programs.csv](equivalent_programs.csv) maps each syntactically enumerated duplicate to its representative. There were 21,384 enumerated programs and 5,901 distinct codebooks.

For the planted control, tiles were generated with `r0=XOR(s, i)`, random eight-bit seeds and the same global scale. Its codebook attained exactly zero error. Another program also attained zero, so the planted mapping was recovered up to a nonunique optimum.

## Storage and limits

Each held-out method stores 4,096 tile bytes. The two-instruction program uses six bytecode bytes under the declared three-byte-per-instruction encoding (opcode and two operand identifiers), plus a four-byte global scale: 4,106 bytes total, or 2.0048828125 bits/weight over this sample. The original 16,384 BF16 weights occupy 32,768 bytes. The scalar method stores four FP16 levels (eight bytes). VQ stores 256 × four FP16 values (2,048 bytes). Decoder code, grammar constants, matrix shapes and the experimental sample manifest are outside these payload counts; all are shared or pre-existing and must be included if deploying a standalone decoder.

This measures weight reconstruction only. It does not establish model perplexity, downstream accuracy, or an optimum outside the finite grammar. Separate synthesized sign, exponent and fraction fields remain untested. The GPU dot products choose seed indices; final errors are accumulated from reconstructed residuals in FP64 to avoid near-zero cancellation. The observed result is negative for this narrow two-instruction grammar: ordinary scalar quantization is better at approximately the same storage rate.

## Reproduce

With the local checkpoint and ROCm device exposed:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.synthesis.miniature \
  --checkpoint data/smollm2-135m.safetensors \
  --out experiments/results/miniature-codebook --device cuda
```

Run time was about 14 seconds after Python startup. Python 3.12.3, NumPy 2.5.5, SciPy 1.18.1, PyTorch 2.9.1+rocm6.4. The search code can also run on CPU for verification. [summary.json](summary.json) and [rate_distortion.csv](rate_distortion.csv) contain machine-readable measurements.
