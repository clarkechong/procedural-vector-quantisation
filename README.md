# Computational codebooks and procedural weights

This repository investigates how compact, inexpensive programs can generate pretrained neural network weights. The existing experiments cover dictionary pursuit, rate–distortion baselines, SeedLM, integer program synthesis, model perplexity, and native decoder benchmarks. The [research proposal](docs/proposal.md) describes the next experiments.

The [SmolLM2-135M results](experiments/results/README.md) index lists the preserved runs, raw measurements, and reports.

## Repository layout

| Directory | Contents |
| --- | --- |
| [`experiments/manifests`](experiments/manifests/) | Checkpoint reader, shared sampling, and provenance conventions |
| [`experiments/target_geometry`](experiments/target_geometry/) | Planned weight-distribution and covariance measurements |
| [`experiments/rate_distortion`](experiments/rate_distortion/) | Dictionary pursuit pilot, SeedLM reproduction, and codebook comparisons |
| [`experiments/generator_algebra`](experiments/generator_algebra/) | Reference integer semantics and dictionary generators |
| [`experiments/synthesis`](experiments/synthesis/) | Program grammar, enumeration, selection, and reports |
| [`experiments/encoding`](experiments/encoding/) | Exact and approximate encoders, packing, and reference decoders |
| [`experiments/evaluation`](experiments/evaluation/) | Model perplexity, activation error, and encoder audits |
| [`experiments/hardware`](experiments/hardware/) | Native SeedLM encoding and CPU/HIP decoder benchmark |
| [`experiments/training`](experiments/training/) | Planned procedural training experiments |
| [`experiments/results`](experiments/results/) | Preserved run configurations, raw metrics, packed samples, plots, and reports |
| [`tests`](tests/) | Deterministic reference and codec tests |

## Set up and run experiments

Run the commands from the repository root. uv uses Python 3.13 by default and installs the versions recorded in `uv.lock`. The base environment covers CPU weight experiments and plotting. The pilot reads local Hugging Face SafeTensors checkpoints, including sharded FP32, FP16, and BF16 checkpoints.

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests
uv run --locked python -m experiments.rate_distortion.pilot --out runs/synthetic --tiles 4 --repeats 2
```

The pilot uses the original BF16 SmolLM2-135M checkpoint. The file checksum is recorded in each run's configuration. [Pinned checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/d6a5589c239236d22370e2126bbe23d4843c47d9/model.safetensors)

```sh
mkdir -p data
curl -fL -o data/smollm2-135m.safetensors \
  https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/d6a5589c239236d22370e2126bbe23d4843c47d9/model.safetensors

uv run --locked python -m experiments.rate_distortion.pilot \
  --checkpoint data/smollm2-135m.safetensors --out runs/smollm2-135m \
  --layers 0 15 29 --tiles 32 --repeats 19 --permutation
```

To repeat the main sweep with Llama 2, point the runner at an authorized local copy of its Hugging Face checkpoint. [Llama-2-7B model access](https://huggingface.co/meta-llama/Llama-2-7b-hf)

```sh
uv run --locked python -m experiments.rate_distortion.pilot \
  --checkpoint /path/to/Llama-2-7b-hf --out runs/llama2-7b \
  --layers 0 15 31 --tiles 32 --repeats 19 --permutation
```

Use `--projections q_proj --layers 15 --blocks --skip-rows --whole-matrix` to exercise exact full-matrix geometry on one projection. Full-matrix runs materialize the dictionary in memory and use randomized truncated SVD. For smaller matrices it is practical to measure all granularities at once.

## Experimental choices

- **Granularity.** Draw up to 32 nonoverlapping tiles uniformly from an aligned grid per projection and size in the pilot. The grid drops incomplete edge tiles. Row samples span the entire projection input dimension. Each matrix contributes equally to the aggregate. Within each matrix, NMSE is the sum of squared residuals divided by the sum of squared original values across sampled tiles.
- **Controls.** Global shuffle draws without replacement from all matrix elements, equivalent in distribution to reading the sampled positions of a uniformly shuffled matrix. Within-tile shuffle preserves each sampled tile's exact multiset. The Gaussian control matches the full matrix mean and standard deviation. The positive control combines a period-four pattern and pseudorandom signs. The mixed control adds that signal at one times the centered tile RMS. Each control has the same geometry as the original samples.
- **Methods.** The mixed dictionary includes constant, affine, XOR, Walsh, periodic, group and seeded sign atoms. Greedy pursuit chooses the atom that reduces actual squared error most after FP16 coefficient rounding. Two ablations restrict it to coordinate atoms or to random atoms. Baselines include one seed with fitted Walsh-modulated pseudorandom basis coefficients, magnitude sparsity, uniform min–max affine quantization, per-tile Lloyd–Max scalar quantization, and truncated SVD. The seeded basis is inspired by SeedLM and uses a different seed search and decoder.
- **Rates.** Report modeled payload bits per weight. A pursuit term costs 8 bits for the atom ID and 16 for its coefficient. Seeded bases cost 8 bits for the seed and 16 per coefficient. Sparse weights cost 16 bits plus the fixed-width position. Scalar quantizers include their FP16 scales and offsets or all FP16 centroids. SVD counts both stored FP16 factors. The experiment assumes the method, setting, shape and generator definitions are shared. Rate budgets use the best measured setting within budget, with a zero reconstruction available at every budget. Actual container size, bandwidth and decode cost are outside this measurement.
- **FFN ordering.** Fit a shared intermediate-neuron permutation by sorting the first principal component of standardized gate, up and down weights from the first quarter of input or output coordinates. Compare with the original and random orders on tiles from the disjoint final three quarters. Verify the permuted SwiGLU FFN numerically with three float64 input vectors. The test fits one ordering heuristic.
- **Inference.** The paired CSV reports null mean minus observed NMSE and the number of matrices with lower error than their own mean null. Monte Carlo ranks are one-sided, unadjusted and exploratory across budgets, methods and block sizes. The broad sweep's 19 random draws give a minimum rank-based p-value of 0.05. Random neuron orders serve as descriptive references for the fitted order.

Each run writes `config.json`, raw setting-level `curves.csv.gz`, aggregated `budget_curves.csv`, `paired_controls.csv` and `rate_distortion.png`.

## Related methods

- SeedLM uses linear-feedback shift-register seeds and low-dimensional coefficients to compress pretrained weights. [SeedLM](https://arxiv.org/abs/2410.10714)
- QTIP combines trellis quantization and incoherence processing with computed codebooks. It is a distinct method from the pilot's scalar quantizers and seeded basis. [QTIP](https://arxiv.org/abs/2406.11235)
- HashedNets trains with hash-based weight sharing. Permute, Quantize and Fine-Tune searches for function-preserving permutations in conjunction with compression. [HashedNets](https://arxiv.org/abs/1504.04788) · [Permute, Quantize and Fine-Tune](https://arxiv.org/abs/2010.15703)

The initial sweep addresses pretrained weight reconstruction. The computational codebook follow-up also measures activation-weighted error and perplexity.

## SeedLM reproduction

The [SeedLM results](experiments/results/seedlm-smollm2/README.md) include the published configurations and follow-up measurements. The SeedLM runner implements the paper's LFSR, normalization, pseudoinverse fitting, and exhaustive seed selection. It also runs paired controls, fitting ablations and a generator and payload design screen. It supports both published geometries and verifies every winning quantized payload through an independent decoder. [SeedLM paper](https://openreview.net/pdf?id=u3TL0qxLWf) · [ArXiv exponent equation](https://arxiv.org/html/2410.10714v2#S3.SS2)

```sh
uv run --locked python -m experiments.rate_distortion.seedlm \
  --checkpoint data/smollm2-135m.safetensors --out runs/seedlm-smollm2 --phase all
```

Use `--phase reference` for the two published configurations and their coefficient and seed-count controls. `--phase ablations` adds the fixed-geometry generator and fitting comparisons. `--phase all` additionally screens LFSR payload configurations and exhaustively evaluates the selected finalists on separate rows.

The reproducible choices for the published recipe are round-to-nearest-even, signed exponent clipping to `[-8, 7]`, and a row-major matrix populated with successor states of the stored seed. The reference bitstream stores the seed, signed exponent and signed coefficients with no per-block padding. This defines a local serialization of the published fields. The alternative `range` and `refine` quantizers use the same payload. `range` chooses the smallest available power-of-two scale that fits all coefficients. `refine` tries three nearby exponents and one integer coordinate-descent pass on the reconstruction objective. Float coefficients serve as an oracle with a larger payload.

## Computational codebook follow-up

The follow-up includes an exact discrete SeedLM oracle, binary and ternary bases, direct and additive seeds, signed permutations, and lattice and trellis baselines. The model run quantizes the seven linear projections in every decoder layer. It evaluates reconstructed weights with the reference Llama implementation on WikiText-2. [Results](experiments/results/README.md#computational-codebook-follow-up)

```sh
VECLIB_MAXIMUM_THREADS=1 uv run --locked python -m experiments.rate_distortion.codebooks \
  --checkpoint data/smollm2-135m.safetensors --out runs/codebooks-smollm2

clang++ -O3 -std=c++17 experiments/hardware/seedlm_decode.cpp -o /tmp/seedlm-decode
/tmp/seedlm-decode 2097152 9 > runs/codebooks-smollm2/cpu_decode.csv
uv run --locked python -m experiments.rate_distortion.codebooks_report runs/codebooks-smollm2
```

The `model` extra installs the dependencies for perplexity and activation error evaluation. Use the dataset's raw WikiText-2 variant and place the two Parquet files and the model's config and tokenizer in one local directory. [Model config revision](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/d6a5589c239236d22370e2126bbe23d4843c47d9/config.json) · [Tokenizer revision](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/tokenizer.json) · [WikiText-2 dataset revision](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1)

```sh
uv sync --locked --extra model
data=/path/to/smollm2-and-wikitext-files
VECLIB_MAXIMUM_THREADS=2 HF_HUB_OFFLINE=1 uv run --locked --extra model python -m experiments.evaluation.seedlm_model \
  --checkpoint data/smollm2-135m.safetensors --data "$data" \
  --out runs/codebooks-smollm2/model --context 1024
VECLIB_MAXIMUM_THREADS=2 HF_HUB_OFFLINE=1 uv run --locked --extra model python -m experiments.evaluation.seedlm_model \
  --checkpoint data/smollm2-135m.safetensors --data "$data" \
  --out runs/codebooks-smollm2/model-indexed --context 1024 \
  --shortlist 12 --neighborhood 256 --methods indexed_refine
```

On a ROCm machine with an MI300X, the HIP build reports device-copy bandwidth and kernel-event timings for materialized decode and an eight-weight dot product. The large input below exceeds the 256 MB last-level cache. [AMD MI300X specifications](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)

```sh
mkdir -p runs/codebooks-smollm2
hipcc -x hip -O3 -std=c++17 --offload-arch=gfx942 experiments/hardware/seedlm_decode.cpp -o /tmp/seedlm-hip
/tmp/seedlm-hip 134217728 9 > runs/codebooks-smollm2/mi300x_decode.csv
```

## Automated generator search

The search builds integer DAGs from the selected operations and literal palette, then scores their signed byte outputs over every available seed and all 16 dyadic scales. The default format stores 12 seed bits and a signed 4-bit exponent per eight weights. The [experiment report](experiments/results/generator-search-smollm2/README.md) gives the grammar boundary, winners, matched-rate comparisons, and decode verification.

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --locked python -m experiments.synthesis.search \
  --checkpoint data/smollm2-135m.safetensors --out runs/generator-search-smollm2

uv run --locked python -m experiments.synthesis.report runs/generator-search-smollm2
uv run --locked python -m unittest discover -s tests
```

Use `--ops` and `--constants` to change the grammar, `--seed-bits` to change the seed budget, and `--max-cost` to set the scalar operation budget per block. Budgets of 33, 41 and 49 operations permit one, two and three generator instructions respectively. The estimate includes conversion of the signed byte to a scaled weight and setup of the shared exponent. The default `--three-limit 240000` uses a bounded search. A value of `0` walks the full three-instruction grammar, which the structural sample estimates at over 100 million live DAGs with the default palette.

## Generator perplexity evaluation

The full-model run applies the frozen synthesized generator and the published 3- and 4-bit SeedLM configurations to all attention and MLP projections in SmolLM2-135M. The reference model evaluates reconstructed weights on WikiText-2. The report includes the seed search audit, packed storage, and reproduction commands. [Perplexity report](experiments/results/generator-model-smollm2/README.md)

Use `--methods synthesized_3` to evaluate the same frozen program at 3 bits per projection weight: a 20-bit seed and a signed 4-bit exponent per eight weights. This expands the seed domain rather than performing a new program search. The encoder indexes unscaled codewords once and queries all 16 scales separately, avoiding an index over 16 million scaled encodings. This index is used only for offline encoding; decoding still executes the generator from the packed seed and exponent.

Use SmolLM2 for fast, controlled comparisons. Use Llama 2 7B for comparisons against the published perplexity numbers. The larger-model guide covers sharded checkpoints, the Linux SeedLM build, optional CUDA offload, and the WikiText-2 evaluation settings. [Llama 2 7B reference run](experiments/evaluation/llama2-7b.md)
