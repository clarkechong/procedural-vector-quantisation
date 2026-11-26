# SeedLM on SmolLM2-135M

The experiment uses the same 21 original BF16 projection matrices as the previous pilot, from layers 0, 15 and 29. It resamples contiguous row spans to align both published block lengths. Each matrix contributes two 192-weight spans to each split. Screening and holdout use disjoint rows, and all codecs at a given geometry see the same held-out vectors and controls. NMSE is residual squared norm divided by weight squared norm, averaged equally across matrices. The controls are one global shuffle sample, one within-block shuffle and one Gaussian draw per matrix. Comparisons with these controls are descriptive. [Checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/d6a5589c239236d22370e2126bbe23d4843c47d9/model.safetensors)

## Published configurations

| Published payload | Weights per block | Coefficients | One seed | 4,095 seeds | 65,535 seeds | Global shuffle | Within-block shuffle | Gaussian |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 bits/weight | 12 | 4 | 0.7324 | 0.1474 | 0.0952 | 0.0952 | 0.0945 | 0.0926 |
| 4 bits/weight | 8 | 3 | 0.7146 | 0.0707 | 0.0370 | 0.0385 | 0.0363 | 0.0367 |

The first three error columns use pretrained weights. The controls use the full search. The 4-bit floating-coefficient oracle achieved 0.6289 NMSE at one seed, close to the isotropic expectation of 0.625, and 0.0091 after exhaustive search. Its rate is higher because the coefficients remain float32. A sharp gain on shuffled and Gaussian data demonstrates the benefit of the candidate search itself. [Paired controls](paired_controls.csv)

Each reference block evaluates all 65,535 nonzero LFSR seeds, builds the normalized basis from the outgoing successor sequence, fits by pseudoinverse, applies the published exponent equation and selects the best reconstruction after rounding. Encoded winners were packed and decoded, and their errors matched the search. The local packing layout, round-to-nearest-even and clipping at the signed exponent endpoints are explicit implementation choices. This is a reconstruction study on SmolLM2. The paper's Llama accuracy, perplexity and FPGA throughput are separate measurements. [Published paper, sections 3.1–3.4 and appendix](https://openreview.net/pdf?id=u3TL0qxLWf) · [Exponent equation](https://arxiv.org/html/2410.10714v2#S3.SS2)

## Experimental variants

The range-fitted exponent reduces pretrained NMSE from 0.0952 to 0.0526 at 3 bits and from 0.0370 to 0.0155 at 4 bits. The basis, search and packed bit widths are identical to the reference. The published exponent scales the largest coefficient near unit magnitude, while the range-fitted exponent uses more of the signed four-bit coefficient range.

| Fitting, quantizer and generator | 3 bits/weight | 4 bits/weight |
| --- | ---: | ---: |
| Published joint fit and exponent | 0.09518 | 0.03698 |
| Greedy fit with published exponent | 0.11390 | 0.04718 |
| Greedy fit with range-fitted exponent | 0.08262 | 0.02756 |
| Joint fit with range-fitted exponent | 0.05258 | 0.01547 |
| LFSR with quantization-aware refinement | 0.05031 | 0.01454 |
| Xorshift with range-fitted exponent | 0.04733 | 0.01389 |
| SplitMix with range-fitted exponent | 0.04854 | 0.01385 |

These rows change one or more encoder settings on the same held-out data and retain the two published payload sizes. The first five rows decode with the same LFSR and packed fields.

The greedy fit visits each basis column at most once. The joint fit estimates the coefficient vector in a single pseudoinverse solve. Both then use the same quantizer and search over the same candidate bases. The seed-count ablation keeps the stored width fixed at 16 bits and searches nested subsets in a deterministic permuted order. The independent-seed ablation splits the same 16 bits among the basis columns and removes tuples containing a zero LFSR state. The shared-seed search includes an equal-candidate-count subset for this comparison.

For the shared versus independent seed ablation at equal candidate counts and with the range-fitted exponent, the shared basis reached 0.0549 at 3 bits and 0.0157 at 4 bits. The tested independent construction reached 0.1862 and 0.0271. This comparison uses 50,625 candidates at 3 bits and 60,543 at 4 bits.

## Design screen

The screen covered 430 LFSR payload configurations and ten additional generator configurations at the published geometries. It varied block length, seed width, latent dimension and coefficient precision. It used four blocks per matrix and at most 1,024 seeds per configuration. One configuration within each target budget advanced to a full search and quantization-aware fit on held-out rows.

| Budget | Actual bits/weight | Block | Seed bits | Coefficients × bits | Seeds searched on holdout | Held-out NMSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.969 | 32 | 12 | 5 × 3 | 4,095 | 0.4557 |
| 2 | 2.000 | 12 | 8 | 4 × 3 | 255 | 0.1997 |
| 3 | 3.000 | 8 | 8 | 4 × 3 | 255 | 0.0652 |
| 4 | 4.000 | 8 | 8 | 5 × 4 | 255 | 0.0218 |

At 3 and 4 bits, the selected 255-seed designs improve on the published exponent recipe and have higher error than the refined 65,535-seed LFSR. They also use more coefficients per weight, shifting work from the offline search to decoding. The best earlier pilot method reevaluated within the 4-bit budget reached 0.3339 NMSE. The best tested scalar baseline within that budget reached 0.0413. [Design space plot](design_space.png) · [Screening measurements](screen.csv)

## Reproducibility and costs

[Configuration and encoding times](config.json) records the input checksum, screen choices and whole-run wall times for each codec and set of fitting rules. Those times include winner verification. [Decode arithmetic](decode_cost.csv) reports the number of generated basis values and multiply-accumulates, with approximate integer operation counts. [Reference CPU decode](cpu_decode.csv) times unpacking and reconstructing equal-length synthetic streams in NumPy on arm64. These CPU measurements depend on the implementation. The different exponent and fitting rules for a fixed codec share its decoder exactly.

[Measurements by matrix](measurements.csv) · [Aggregated results](summary.csv) · [Paired controls](paired_controls.csv) · [Scalar and earlier pilot baselines](scalar_and_pilot_baselines.csv) · [SeedLM figure](seedlm.png)
