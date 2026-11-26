# SmolLM2-135M results

Each directory records the inputs and preserves the outputs for a single study. Raw tables, packed samples, plots and configurations are kept together. Start new runs under the ignored `runs/` directory. When publishing a result, place the configuration, raw measurements and report in a new directory here.

| Study | Preserved runs and reports |
| --- | --- |
| Procedural dictionary pursuit | [Main sweep](smollm2-135m/), [sensitivity](smollm2-135m-sensitivity/), [whole matrix](smollm2-135m-whole/), [FFN ordering](smollm2-135m-ordering/) |
| SeedLM reproduction | [SeedLM report](seedlm-smollm2/README.md) |
| Computational codebooks | [Study report](#computational-codebook-follow-up) and [raw outputs](codebooks-smollm2/) |
| Generator synthesis | [Search report](generator-search-smollm2/README.md) |
| Generator perplexity | [Evaluation report and partial runs](generator-model-smollm2/README.md) |
| Laptop model evaluation | [Completed synthesized and SeedLM runs, including 3-bpw synthesis](laptop-rtx3050/README.md) |

The input is the original BF16 checkpoint. The main sweep covers seven projections at each of layers 0, 15 and 29, five tile sizes from 4×4 to 64×64, and full rows. It draws up to 32 tiles per matrix and scale and uses 19 shuffle draws per null. [Checkpoint and SHA-256](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/d6a5589c239236d22370e2126bbe23d4843c47d9/model.safetensors)

## Main sweep

The mixed procedural dictionary found a small position-dependent signal at some scales. At 2 bits/weight, the original 8×8 tiles had 0.6848 NMSE versus 0.6896 after global shuffling and 0.6914 after shuffling only within tiles. The pooled Monte Carlo rank was 0.05 for each null, the smallest attainable with 19 draws. These are exploratory comparisons across many rates and geometries. At 16×16, the original and shuffled results were within 0.0008 NMSE.

| Granularity | Mixed dictionary | Global shuffle | Within-tile shuffle | Best baseline on original weights |
| --- | ---: | ---: | ---: | --- |
| 4×4 | 0.6591 | 0.6625 | 0.6625 | 0.6780, sparse |
| 8×8 | 0.6848 | 0.6896 | 0.6914 | 0.3814, Lloyd–Max |
| 16×16 | 0.8599 | 0.8603 | 0.8607 | 0.4026, Lloyd–Max |
| 32×32 | 0.9645 | 0.9646 | 0.9646 | 0.4186, Lloyd–Max |
| 64×64 | 0.9909 | 0.9911 | 0.9911 | 0.4216, Lloyd–Max |
| Full row | 0.9540 | 0.9543 | 0.9546 | 0.3975, Lloyd–Max |

Table entries are mean NMSE at a budget of 2 bits/weight. Lower is better. The 4×4 comparison favored the dictionary at this budget because of the available payload configurations at this size. Across granularities, a conventional baseline had lower mean NMSE than the best procedural setting at all eight tested budgets. At 2 bits/weight the best values were 0.3814 with 8×8 Lloyd–Max and 0.6591 with 4×4 pursuit. [Rate–distortion plot](smollm2-135m/rate_distortion.png) · [Budget curves](smollm2-135m/budget_curves.csv) · [Paired controls](smollm2-135m/paired_controls.csv)

## Additional checks

- **Larger pursuit budget.** At 16×16, 64 allowable atoms and an independent tile sample, the original, globally shuffled and within-tile shuffled results were 0.8488, 0.8502 and 0.8496 NMSE at 2 bits/weight. Lloyd–Max reached 0.4035. At 4 bits/weight, the mixed dictionary reached 0.8359 and Lloyd–Max reached 0.0423. The planted control reached 4.9 × 10⁻⁷ NMSE at 2 bits/weight. Injecting the signal into pretrained weights yielded 0.4258 NMSE at the same budget. [Sensitivity curves](smollm2-135m-sensitivity/budget_curves.csv)
- **Whole matrix.** Layer 15's 576×576 query projection gave 0.99983 NMSE with 16 global atoms versus 0.99990 after a global shuffle. The 16-atom payload was about 0.00116 bits/weight, and the curve plateaus at that model capacity. The whole-matrix test covers one projection. [Whole-matrix curves](smollm2-135m-whole/budget_curves.csv)

- **Shared FFN permutation.** In the main sweep the fitted ordering helped at 8×8 and slightly worsened the sampled result at 16×16. Evaluating all 2,592 held-out 16×16 tiles per FFN matrix reduced mean NMSE from 0.8596 to 0.8577 at 2 bits/weight, improving 8 of 9 matrices. Random order averaged 0.8594. The largest relative difference between original and permuted FFN outputs in these runs was 1.74 × 10⁻¹⁵. [Full holdout comparisons](smollm2-135m-ordering/paired_controls.csv)

## Measurement limits

The experiment evaluates fixed generators and a single ordering heuristic on one checkpoint. Llama-2-7B remains untested because its original checkpoint was unavailable locally and the official download is gated. Weight NMSE and modeled payload sizes are the measured outcomes. The computational codebook follow-up below covers activation error, perplexity and local decoder cost. [Llama-2-7B access](https://huggingface.co/meta-llama/Llama-2-7b-hf)

## Computational codebook follow-up

The weight experiments use the same BF16 checkpoint and 21 projections as the earlier SeedLM run. Each matrix contributes equally. The subspace and vector comparison uses one held-out 192-weight span per matrix. The discrete oracle uses the first two eight-weight blocks of each span. The rotated baselines pick an FP16 scale using separate calibration rows and use a deterministic signed 64-point Hadamard transform. [Sampling configuration](codebooks-smollm2/sampling.json) · [Reproduction commands](../../README.md#computational-codebook-follow-up)

### Exact fitting

| 4-bit basis | Paper equation | Refined | Exact with signed 4-bit exponent | Unbounded integer exponent | Real scale with integer coefficients | Floating projection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense LFSR | 0.037864 | 0.014379 | 0.014334 | 0.014334 | 0.010080 | 0.008674 |
| Binary LFSR | 0.062632 | 0.019219 | 0.019105 | 0.019014 | 0.014793 | 0.013745 |
| Ternary LFSR | 0.044088 | 0.013899 | 0.013859 | 0.013836 | 0.010475 | 0.009120 |

For each seed, the continuous projection supplies a lower bound. A surviving seed enumerates all 4,096 coefficient triples. For each triple, the error is quadratic in the positive scale, so checking the two adjacent dyadic scales with exponent clipping finds the discrete optimum. Search starts from the refined packed codeword. The dense oracle needed a median of three seeds per block after evaluating projection bounds over all 65,535 seeds. The exact dense encoder improved refined NMSE by 0.32% relative. The exponent range made no measurable difference on these blocks. Relaxing the power-of-two scale accounted for about three quarters of the remaining gap to the floating projection. The free real scale and unbounded exponent are diagnostic bounds. I found no public author encoder linked from Apple's paper page or the conference material, so the paper-equation comparison uses the stated method and local conventions. [Exact results and packed words](codebooks-smollm2/oracle.csv) · [Dense binary payload](codebooks-smollm2/oracle_lfsr.bin) · [SeedLM methodology](https://arxiv.org/html/2410.10714v2#S3.SS2) · [Apple publication](https://machinelearning.apple.com/research/seedlm-compressing)

### Coverage, rate and decode cost

| 4-bit codec | Held-out NMSE | Apple M1 materialized decode, Gweights/s |
| --- | ---: | ---: |
| Dense LFSR, paper equation | 0.03801 | 1.001 |
| Dense LFSR, refined | 0.01448 | 1.001 |
| Binary LFSR, refined | 0.01903 | 1.463 |
| Ternary LFSR, refined | 0.01400 | 0.928 |
| Xorshift, refined | 0.01312 | 0.448 |
| SplitMix, refined | 0.01332 | 0.490 |
| Direct seed, six weights | 0.01390 | |
| Additive seeds, eight weights | 0.02217 | |
| Signed permutation, eight weights | 0.03156 | |
| Rotated residual E8-shaped lattice | 0.01978 | |
| Rotated 1MAD trellis | 0.00967 | |

The ternary basis uses LFSR bit 0 for the sign and bit 7 to select zero. It needs three masked additions or subtractions per weight plus scaling. It produced approximately 64,982 distinct rounded orthogonal projectors from 65,535 seeds, compared with 49,187 for the binary basis and 128 for the signed Hadamard construction. At full search the paired ternary-minus-dense mean difference was −0.00048 NMSE with a matrix-bootstrap interval of [−0.00109, +0.00011]. The two variants were close on this checkpoint. The independent random-subspace reference predicts a mean minimum of 0.007915 at 65,535 draws, while floating dense LFSR reached 0.009075 and floating ternary LFSR reached 0.009135. The reference assumes independent uniformly distributed subspaces. Its expected minimum is the integral of the beta distribution's survival function raised to the seed count. The Gaussian rate–distortion line is an asymptotic comparison under an IID Gaussian source assumption. [Subspace curves](codebooks-smollm2/subspaces.csv) · [Projector and theory data](codebooks-smollm2/geometry.csv) · [Coverage and decoder plot](codebooks-smollm2/codebook_coverage.png)

The direct codec exhaustively searches a 20-bit seed namespace and a 4-bit dyadic exponent without fitted coefficients. The additive codec uses two 14-bit seeds and a 4-bit exponent, with alternating nearest-vector search and four initial choices. The signed permutation stores a 16-bit rank, 4-bit generated magnitude template, eight signs, and a 4-bit exponent. The shaped lattice uses generated E8 shells and a deterministic choice of padding, with a second lattice index for the 4-bit residual variant. The trellis uses QTIP's 16-bit scalar 1MAD map, exact open-boundary Viterbi from a known zero state, and a fixed signed Hadamard rotation for the rotated variant. The full QTIP and QuIP# pipelines also use calibration objectives and transforms beyond this local comparison. The lattice and trellis store one additional 16-bit scale per matrix beyond the listed payload rates. [Vector runs](codebooks-smollm2/other_codebooks.csv) · [Rotated baselines](codebooks-smollm2/rotated_baselines.csv) · [Rate–distortion plot](codebooks-smollm2/rate_distortion.png) · [QTIP code](https://github.com/Cornell-RelaxML/qtip) · [QuIP# lattice](https://proceedings.mlr.press/v235/tseng24a.html)

The CPU timings are medians of nine passes over 2,097,152 packed blocks on one Apple M1 core. The three LFSR decoders generate four feedback bits at a time. Dense decoding extracts full states, binary decoding extracts signs, and ternary decoding extracts signs and zero masks. The fused microbenchmark returns one dot product per eight weights. The HIP benchmark uses the same decoders and includes a device-copy reference. For a 4-bit payload, AMD’s 5.3 TB/s peak corresponds to 10.6 trillion weights/s for input alone or 1.18 trillion weights/s if each weight also produces a 32-bit stored output. The workspace has no ROCm toolchain or MI300X, so the HIP build and bandwidth measurements are pending. [CPU timings](codebooks-smollm2/cpu_decode.csv) · [Measured CPU Pareto frontier](codebooks-smollm2/pareto_cpu.csv) · [Benchmark source](../hardware/seedlm_decode.cpp) · [AMD MI300X specifications](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)

### Full-model perplexity and activation-weighted error

All model runs scored 304,680 next-token targets from the full raw WikiText-2 test split with 1,024-token nonoverlapping contexts. The compressed runs replace all 210 attention and MLP projection matrices, covering 106,168,320 parameters, or 78.9% of the model. The 4-bit SeedLM projections require 53,084,160 packed bytes. Including BF16 embeddings and normalizations gives an effective 6.53 bits per parameter. The 4.25-bit scalar rate includes a 16-bit scale per 64 weights, for 6.73 bits per parameter overall. Inference loads the packed reconstructions into the same reference float32 model. [WikiText-2 revision](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1)

| Model | Projection bits/weight | Full projection NMSE | Mean activation NMSE | Test perplexity |
| --- | ---: | ---: | ---: | ---: |
| Original BF16 checkpoint | 16.00 | 0 | 0 | 17.49 |
| Group-64 INT4 | 4.25 | 0.01272 | 0.01188 | 25.03 |
| Dense LFSR, indexed refinement | 4.00 | 0.02005 | 0.02100 | 41.07 |
| Dense LFSR, 255-seed refinement | 4.00 | 0.08432 | 0.08264 | 24,477 |
| Dense LFSR, 255-seed paper equation | 4.00 | 0.14416 | 0.13961 | 2,998,974 |

The fixed-prefix runs screen the same 255 LFSR seeds and quantize the eight best projections per block. The indexed run builds 7,168 buckets from all 65,535 seeds using the signs and positions of the two largest weight magnitudes. It ranks 256 candidates from the matching bucket and applies refinement to the best 12. The block payload and decoder are identical between dense LFSR runs. Local single-run encoding times were 59 seconds for the fixed paper equation, 99 seconds for fixed refinement, and 430 seconds for indexed refinement, including the index build. These timings came from runs with some concurrent local work.

Activation NMSE compares each reconstructed linear projection with its original output on the same inputs from 512 validation tokens. The metric averages the 21 projections sampled at layers 0, 15 and 29. Fixed refinement improved measured output NMSE on all 21 projections compared with the fixed paper equation. Indexed refinement improved all 21 again. The full-model result shows a large gain from input-dependent seed search at the same rate and decoder. The INT4 comparison used an extra 0.25 bits per projection weight and had lower perplexity. [Combined model metrics](codebooks-smollm2/model_quality.csv) · [Fixed-search runs](codebooks-smollm2/model/perplexity.csv) · [Indexed run](codebooks-smollm2/model-indexed/perplexity.csv) · [Fixed-search activation errors](codebooks-smollm2/model/activation_error.csv) · [Indexed activation errors](codebooks-smollm2/model-indexed/activation_error.csv)

## Automated generator search

The local search covers one to three unsigned integer instructions, with a 16-bit payload per eight SmolLM2 weights at the primary rate. The experiment report contains the scope of enumeration, frozen programs, matched-rate baselines, and rate–distortion curves. [Generator search report](generator-search-smollm2/README.md)

## Generator perplexity evaluation

The model-quality experiment evaluates the frozen three-instruction generator and the published SeedLM configurations on the same SmolLM2 checkpoint and WikiText-2 test split. [Full-model perplexity report](generator-model-smollm2/README.md)
