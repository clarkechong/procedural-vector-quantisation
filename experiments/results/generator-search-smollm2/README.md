# Automated generator search on SmolLM2-135M

This experiment searches coefficient-free weight generators at 2 bits per weight. Each eight-weight block stores a 12-bit seed and a signed 4-bit exponent. One program is shared across all sampled matrices.

## Results at 2 bits per weight

| Generator or baseline | Holdout NMSE ↓ | Unique scaled codewords | Estimated scalar operations per block |
| --- | ---: | ---: | ---: |
| Best 1-op | 0.55595 | 4,081 | 33 |
| Best 2-op | 0.23622 | 60,332 | 41 |
| **Best 3-op** | **0.19910** | **65,473** | **49** |
| Direct LFSR | 0.38908 | 65,521 | |
| Direct Xorshift | 0.26730 | 65,521 | |
| Direct SplitMix | 0.28549 | 65,536 | |
| LFSR, calibrated amplitude | 0.28050 | 65,521 | |
| Xorshift, calibrated amplitude | 0.19408 | 65,521 | |
| SplitMix, calibrated amplitude | 0.18807 | 65,536 | |
| Refined SeedLM | 0.21658 | | |
| Lloyd–Max | 0.17292 | | |
| Explicit Gaussian codebook | 0.16861 | 65,536 | |
| Trained VQ | 0.16449 | 65,536 | |

The three-instruction winner achieved 8.1% lower pooled NMSE than matched-rate Refined SeedLM. The paired 95% matrix-resampling interval for the absolute difference was [−0.03753, +0.01497]. Calibrated Xorshift and SplitMix had lower point estimates. Their respective intervals for synthesized minus baseline were [−0.01492, +0.03620] and [−0.00742, +0.03925]. Trained VQ had the lowest pooled NMSE and used a shared 32 KiB lookup table. The earlier 0.1997 SeedLM measurement used 12-weight blocks and a different holdout. [Paired comparisons](paired_comparisons.csv)

| Frozen winner | Program | Dependency depth | Peak live values |
| --- | --- | ---: | ---: |
| 1-op | `SHL(seed, index)` | 1 | 2 |
| 2-op | `ROTL(seed, SHR(seed, index))` | 2 | 2 |
| 3-op | `x = SHR(MUL(seed, 0x9e3779b9), index)`, then `SHR(x, x)` | 3 | 2 |

The three-instruction codebook had 4,093 distinct unscaled vectors and effective rank 7.95 out of 8. It used 1,374 seeds on the final 2,688 blocks. The two-instruction winner reached effective rank 7.90 with 3,791 distinct unscaled vectors. The single-instruction winner can use only the seed's low byte, so its distortion plateaus at 1.5 bits per weight. With a 16-bit seed at 2.5 bits per weight, the frozen three-instruction program reached 0.10997 NMSE. The explicit trained VQ curve ends at 2 bits per weight because the 65,536-centroid setup exceeds the calibration sample. [Holdout scores](holdout.csv) · [Diagnostic distributions](diagnostics.json) · [Rate–distortion plot](rate-distortion.png)

## Scalability on Apple M1

| Instructions | Live DAGs visited | Distinct programs scored | Wall time | Exact scoring time |
| --- | ---: | ---: | ---: | ---: |
| 1, exhaustive | 1,166 | 12 | 0.028 s | 0.025 s |
| 2, exhaustive | 324,148 | 5,019 | 12.0 s | 9.9 s |
| 3, bounded | 240,000 | 21,508 | 51.0 s | 43.9 s |

The structural sample estimates 113.3 million live three-instruction DAGs before dependency and semantic filtering, with a sample standard error of 8.2 million. Within the bounded run, 18,624 DAGs came from promising validation-independent screen prefixes and 221,376 from the permuted prefixes. On 42 blocks, exact nearest-codeword search took a median of 2.41 ms with 4,096 seeds and 37.5 ms with 65,536 seeds. Timings use a single BLAS thread and an already generated codebook for the standalone search benchmark. MI300X measurements were deferred. [Enumeration measurements](enumeration.csv) · [Growth estimate](structural_scaling.csv) · [Search timings](search_scaling.csv)

## Search and evaluation

- **Grammar.** A program is a DAG of at most three unsigned 32-bit instructions drawn from ADD, SUB, MUL, XOR, AND, OR, NOT, SHL, SHR, ROTL, and POPCOUNT. Math wraps modulo 2³² and shift or rotate counts use their low five bits. The operands are seed, index, one of 20 literals, or any previous result. Equivalent one-step literal derivations share a representative. The low byte of the final result is interpreted as a signed integer and multiplied by the stored power of two.
- **Structural search.** Commutative operand permutations, redundant identities and dead instructions are removed. Programs with no seed or no index dependence are excluded from ranking and represented by separate controls. Candidate outputs are fingerprinted at fixed seed values and all eight indices. A fingerprint match is verified over all 4,096 seeds before a program is discarded. The one- and two-instruction walks cover the canonical grammar. The three-instruction run extends the 64 strongest two-instruction screen candidates first, then draws from a deterministic permutation of the other prefixes.
- **Exact encoding.** The residual squared is quadratic in the scale. For a codeword `c` and target `w`, the continuous optimum is their dot product divided by the codeword norm squared. Between adjacent scales `t` and `2t`, the decision boundary is `1.5t`. This gives the minimizing signed exponent directly after clipping to `[-8, 7]`, including zero vectors and nonpositive fits. Each scored generator considers every seed. A brute-force regression test enumerates the full 65,536 encodings and checks the exact minimum.
- **Data.** The seven attention and MLP projections at each of layers 0, 15 and 29 provide 10,752 calibration blocks, 42 search blocks, 1,344 validation blocks, and 2,688 final holdout blocks. The search blocks are two per matrix from the dedicated screening rows. An additional 294 blocks from those rows are available for search timings. Splits use disjoint rows and exclude the rows previously used in the SeedLM and codebook sampling runs. The search retains up to 64 candidates per instruction count. The validation winner at each count is then frozen across all plotted rates. NMSE is total squared residual divided by total target energy over the same blocks for every codec. [Run configuration](config.json) · [Sample manifest](samples.json) · [Validation rankings](validation.csv)
- **Cost model.** An instruction costs one abstract scalar operation. Each weight adds three operations for signed byte extraction, conversion to floating point, and multiplication. The exponent scale adds one operation per block. Depth and peak live inputs or temporaries come from the DAG. Bitfield unpacking, loop overhead, integer instruction latency, compiler scheduling, and hardware throughput are outside this estimate.
- **Baselines.** The direct LFSR, Xorshift and SplitMix encoders use the same signed-byte and exponent format. A second comparison lets validation choose an arithmetic shift of 0–5 on each PRNG's signed byte. Lloyd–Max uses fitted FP16 levels per matrix, with 168 shared bytes at the primary rate. Refined SeedLM selects from matched-rate LFSR configurations with up to four bases on validation and exhausts the retained configuration's valid seeds using the existing coefficient refinement rule. The explicit Gaussian and trained VQ codebooks store eight signed bytes per seed, or 32 KiB each at the primary rate, and read one codeword per block. The trained codebook uses spherical Lloyd iterations on calibration directions and a shared exponent on each encoded block. [Baseline selection](baseline_validation.csv) · [SeedLM paper](https://openreview.net/pdf?id=u3TL0qxLWf)

## Outputs and reproduction

- [Rate–distortion and cost plot](rate-distortion.png) · [Holdout scores and storage](holdout.csv) · [Per-matrix results](by_projection.csv) · [Paired comparisons](paired_comparisons.csv)
- [Enumeration counts and runtime](enumeration.csv) · [Structural growth estimate](structural_scaling.csv) · [Exact search timing](search_scaling.csv) · [All scored programs](screen.csv)
- [Codebook diagnostics](diagnostics.json) · [Frozen winner definitions and payload hashes](winners.json) · [Reproduction commands](../../../README.md#automated-generator-search)

The diagnostic file records covariance, effective rank, output histogram, distance quantiles and seed counts for validation finalists and direct holdout codebooks. Each winning 2-bpw program has a 5,376-byte packed payload for all 2,688 final blocks. The independent scalar interpreter verifies the complete reconstructed weight arrays at every plotted rate. The two non-byte-aligned rates use a contiguous bitstream with padding only at the end of the stream.
