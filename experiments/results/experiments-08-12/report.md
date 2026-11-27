# Experiments 08–12: execution report

27 November 2026. Repository revision `9043d854498a95c33e0fb254e0a1bfd02faf9cd4` plus the working-tree experiment code. PyTorch `2.9.1+rocm6.4`, NumPy `2.5.5`, SciPy `1.18.1`, Transformers `5.24.1`, and PyArrow `26.0.1` were used on the host MI300X. All numerical results below are local adaptations unless explicitly identified as a reference component check. The image test sets and SmolLM2 test tokens were used in earlier work, so they are evaluation data, not a blind final holdout.

## 08: QTIP components and full-model controls

The independent 16-state-bit, 2-bpw, 256-dimensional Gaussian encoder now follows the pinned upstream two-pass tail-biting transition. Its path obeys the circular boundary; a small exhaustive fixture verifies the best path for a fixed overlap. On 256 IID Gaussian vectors, **1MAD MSE was 0.06887 ± 0.00087** and **3INST MSE was 0.06879 ± 0.00084** after dividing the 3INST codebook by its RMS of 1.24371. The intervals are 1.96 standard errors over vectors. Both agree with the [paper's rounded 0.069 Table 1 value](https://openreview.net/pdf?id=7sdkLVuYCU). Raw, unnormalised 3INST gave 0.07477; it is retained in the raw run. This is an independent algorithm recreation, not a tensor-by-tensor run of the upstream CUDA implementation. See `../reference/qtip-gaussian-tailbite/` and `../reference/qtip-gaussian-tailbite-normalized/`.

For synthetic 512 × 64 calibration activations, the covariance implementation agreed with explicit outer-product accumulation to `1.44e-15`. A randomized 64D Hadamard transform was invertible to `2.22e-15`, and its transformed covariance matched the analytical transform to `9.99e-16`. The independent 16-column block LDL factors matched the pinned upstream `block_LDL` tensors exactly; reconstruction error was `4.44e-16`. On **actual SmolLM2 layer-15 down-projection activations** from 512 calibration tokens, the 1,536D Hessian and pinned upstream Hadamard source were exercised: transform versus explicit matrix differed by `2.00e-15`, transformed Hessian by `2.22e-16`, and block LDL reconstruction by `3.68e-16`. Upstream RHT orthogonality error was `7.28e-8`, reflecting its FP32 fixed Hadamard submatrix. See `../reference/qtip-pipeline-components/` and `qtip-smollm-components/`. These are mathematical component checks. Official full-projection BlockLDLQ parity, compressed QTIP export, and the eight full-model A–H ablations are **not completed**.

A **bounded eight-condition local ablation** used the first 16 output rows × 256 input columns of that SmolLM2 projection, 512 calibration and 512 subsequent validation tokens, 16-state-bit tail-biting 2-bpw tiles, one frozen matrix RMS scale, and identical tile layout for both maps. The table reports submatrix output NMSE on the validation activations. The circular trellises, scale, and RHT signs were written to eight `.bin` files: 1,028 bytes without RHT and 1,062 bytes with it. Independent unpacking reconstructed every transformed submatrix exactly (`decoder_max_abs = 0`). The stored rates count all those bytes. See `../reference/qtip-submatrix-ablation/`.

| Generator | RHT | Block LDL compensation | Output NMSE | Stored bpw |
|---|---|---|---:|---:|
| QTIP 1MAD | No | No | 0.0852 | 2.008 |
| QTIP 1MAD | No | Yes | 0.0962 | 2.008 |
| QTIP 1MAD | Yes | No | 0.0682 | 2.074 |
| QTIP 1MAD | Yes | Yes | **0.0615** | 2.074 |
| Multiplier-120 | No | No | 0.1378 | 2.008 |
| Multiplier-120 | No | Yes | 0.1513 | 2.008 |
| Multiplier-120 | Yes | No | 0.1255 | 2.074 |
| Multiplier-120 | Yes | Yes | 0.1024 | 2.074 |

This local fixture confirms that the selected QTIP map benefits from RHT plus compensation and outperforms the multiplier-120 map under the matched layout. Its 16 × 256 crop, scale convention, and Python trellis encoder are not the official full projection or model pipeline. No perplexity inference follows from its output NMSE.

The model-level scalar controls used all 210 SmolLM2 projections and 2,032 next-token targets, substantially more than the previous 127-target smoke. The checkpoint was evaluated in FP32 with context 128. The standard zero-inclusive positive controls were:

| Control | Packed projection bpw | Weight NMSE | Perplexity |
|---|---:|---:|---:|
| Dense | 16.000 | 0 | 36.07 |
| INT8, per-output-channel scale | 8.047 | 0.0000731 | 36.37 |
| INT4, group-64 scale | 4.500 | 0.01272 | 53.69 |

The initial one-scale-per-matrix odd-level controls failed at roughly 10–12 million perplexity even at INT8; their rows remain in `../activation_ptq/int_controls/`. The grouped controls are in `../activation_ptq/int_controls_grouped/`. The INT4 control is a simple uniform quantizer, not GPTQ or SmoothQuant. There is no new model-wide 2-bpw QTIP result.

## 09: low-bitrate shadow-weight QAT

The existing 784–256–128–10 MLP, stratified 55k/5k training/validation split, official 10k test split, paired seeds 11/17/23, Adam, and frozen initial layer RMS scales were reused. The 20-epoch critical-rate run used the same 256-example batch size as Experiment 05. Values are mean test accuracy across three seeds; the earlier exposed test split makes them exploratory. See `../procedural_training/low_rate_20epoch/`.

| Representation | Actual bits per linear weight, including biases/scales | MNIST | Fashion-MNIST |
|---|---:|---:|---:|
| Scalar 2-bit | 2.054 | 94.76% | 85.15% |
| Binary scalar | 1.054 | 90.79% | 82.61% |
| Multiplier-120 procedural, 1 payload bpw | 1.054 | 91.82% | 83.99% |
| Multiplier-120 procedural, 0.5 payload bpw | 0.554 | 89.09% | 81.32% |
| Hash32 procedural, 0.5 payload bpw | 0.554 | 89.80% | 81.26% |
| Multiplier-120 procedural, 0.25 payload bpw | 0.304 | 85.86% | 78.45% |
| Rank-2 dense factorization | 0.480 | 69.59% | 72.34% |

The broader five-epoch screening results (same three seeds) were:

| Method | Actual bpw | MNIST | Fashion-MNIST |
|---|---:|---:|---:|
| Dense | 32.054 | 97.37% | 86.62% |
| Scalar INT4 | 4.054 | 94.97% | 84.56% |
| Scalar INT2 | 2.054 | 93.63% | 83.16% |
| Ternary | 1.639 | 92.73% | 83.08% |
| Binary | 1.054 | 90.55% | 81.80% |
| Multiplier-120, 4/2/1.5/1/0.5/0.25 payload bpw | 4.054 / 2.054 / 1.554 / 1.054 / 0.554 / 0.304 | 94.73 / 93.98 / 92.85 / 91.10 / 88.74 / 84.53% | 84.37 / 83.51 / 83.58 / 82.92 / 80.50 / 77.60% |
| QTIP 1MAD, 2/1/0.5 payload bpw | 2.054 / 1.054 / 0.554 | 94.26 / 90.56 / 87.74% | 84.18 / 81.91 / 79.20% |
| Hash32, 1/0.5 payload bpw | 1.054 / 0.554 | 91.65 / 89.53% | 82.76 / 80.20% |
| Random explicit book, 0.5 payload bpw | 1.112 | 89.43% | 79.56% |
| Rank 1/2/4/8 factorization | 0.267 / 0.480 / 0.905 / 1.757 | 33.67 / 64.76 / 86.24 / 93.82% | 36.59 / 68.57 / 80.52 / 82.97% |

The 1-bpw procedural model beat binary scalar at the same storage by 1.03 points on MNIST and 1.38 on Fashion-MNIST, but remained below scalar 2-bit by 2.94 and 1.16 points. The tested 0.5-bpw maps were worse. The 5-epoch sweep additionally includes dense, scalar INT4, ternary, QTIP 1MAD at 1/2 bpw, multiplier-120 at 1.5/2/4 bpw, Hash32, random explicit codebooks, and rank-1/2/4/8 networks. The random explicit codebook costs 1.112 effective bpw at a nominal 0.5 payload bpw once its FP32 table is counted, so it is not an equal-storage comparison. See `../procedural_training/low_rate_qat/` and `../procedural_training/low_rate_controls/`. These fixed tile books have no trellis boundary bits; each independent 8-bit seed is stored as a byte. Rank-2 at 0.480 bpw is a near-budget low-rank control for the 0.554-bpw procedural model, but no architecture search was performed.

## 10: sustained latent-seed surrogate training

All methods use the actual generated weights in the forward pass. The exact discrete-Jacobian backward computes both forced values of each of eight seed bits; the Bernoulli method differentiates the exact expectation over all 256 codewords; the Boolean method truncates the codebook's multilinear extension to degree two. The same 55k training images, three seeds, five epochs, initial codebook projection, and `Adam lr=0.01` were used. The continuous latent parameter is thresholded for the forward pass. See `../procedural_training/latent_qat/`, `latent_bernoulli/`, and `latent_boolean/`.

| 2-bpw method | MNIST test | Fashion-MNIST test |
|---|---:|---:|
| Shadow-weight procedural QAT, five epochs | 93.98% | 83.51% |
| Discrete Jacobian | 92.08% | 83.34% |
| Exact Bernoulli expectation | 92.76% | 83.27% |
| Boolean degree-two surrogate | 11.35% | 9.91% |

The failed Boolean runs are retained. Discrete Jacobian and Bernoulli do train full networks over multiple epochs; neither beats shadow-weight QAT on both tasks at this setting. At 0.25 payload bpw, 32-weight tiles have 8 latent bits and **59,082 continuous trainable parameters including biases**, versus 235,146 for the shadow method. Adam's two moment arrays therefore require 472,656 versus 1,881,168 bytes for those parameters alone, a 3.98× reduction. The five-epoch latent result was 82.82% MNIST and 75.48% Fashion-MNIST. At 20 epochs it reached **83.25% and 76.22%**, versus **85.86% and 78.45%** for the same-rate shadow QAT. This comparison excludes generated weights, gradients, activations, and workspaces; it is a parameter-state calculation, not measured peak GPU memory. See `../procedural_training/latent_025/` and `latent_025_20epoch/`.

## 11: inverse scaling

An MLP bit predictor with fixed 16D input and 256–256 hidden layers trained on 20,000 independent planted noisy codewords per condition, and was tested on 2,000 independent targets. Its thresholded seed plus all one-bit neighbors used B+1 candidate evaluations. An exhaustive nearest-codeword oracle scored 64 selected test targets per condition. See `../learned_optimizer/inverse_scaling/`.

Independent bit extraction at 12 and 16 seed bits reached the exact oracle on all 64 targets. Linear Boolean mixing and Hash32 did not scale under the same budget: at 16 bits, exact-seed hits were 4.7% and 0%, versus 0% for the random explicit codebook. Their mean reconstruction regrets per output were 0.80, 1.01, and 1.00 respectively. At 20 bits, 16 outputs do not identify every input bit even for the bit-extraction control; zero regret on that row does not imply seed recovery. This is a synthetic planted-seed inverse, not the held-out MNIST gradient-trace task or a full learned optimizer. The random explicit codebook's storage is counted. There is no demonstrated runtime or quality scaling advantage over exhaustive search.

## 12: small TinyStories transformer

A byte-level 4-layer, 128-hidden, 4-head, 512-FFN, context-256 decoder was trained from scratch on a deterministic 20,000-story sample from the first public TinyStories training shard. Validation used 1,000 stories from the separate official validation shard. The experiments used one seed, batch 16, and AdamW. Byte-level text generation after this short run is still visibly incoherent; cross-entropy is the primary measurement. See `../procedural_training/tinystories_transformer_extended/` and `tinystories_transformer_1600/`.

At 1,200 matched from-scratch updates, validation cross-entropy was 2.055 dense, 2.196 scalar 2-bit, 2.200 procedural 2-bit, and 2.260 procedural 1-bit. Directly compressing the 1,200-step dense model gave 2.348 at procedural 2-bit and 2.698 at procedural 1-bit. Another 400 compression-aware steps improved those models to 1.971 and 2.090, respectively. At **1,600 total matched updates**, the from-scratch controls reached **1.821 dense, 2.059 scalar 2-bit, 2.046 procedural 2-bit, and 2.198 procedural 1-bit**. Fine-tuning the dense 1,200-step model for another 400 steps therefore beat scratch procedural training at the same 1,600 total steps, but remained worse than dense training.

The discrete-Jacobian latent-seed transformer also trained from scratch at 2 payload bpw, reaching **2.336 validation cross-entropy after 1,600 steps**. That is much worse than 2.046 for shadow-weight procedural QAT at the same steps. Its deployment payload decoded with zero logit difference in the independent CPU check. See `../procedural_training/tinystories_latent2_1600/`.

The 786,432 quantized projection weights use 2.00065 or 1.00065 actual bits per projection weight including 16 FP32 scales. Dense embeddings, layer norms, output head, and biases contribute 105,216 FP32 parameters. Actual packed payloads are 617,536 bytes at 2 bpw and 519,232 bytes at 1 bpw, versus 3,566,592 bytes for the full dense FP32 state. These sizes count every stored parameter, scale and bias in a fixed architecture; they exclude the separate human-readable JSON layout manifest and decoder code. Independent decoding of the binary payloads reproduced logits within `4.3e-6`. See `export_validation.json` and the `.bin` payloads. This is an initial model experiment on one shard, not a converged language-model benchmark or a claim of coherent generation.

## Remaining gates

The requested full SmolLM2 QTIP A–H model ablation and complete upstream intermediate parity are still open; the 16 × 256 local ablation and payload do not substitute for full-projection BlockLDLQ and model-level compressed export. The current Python tail-biting encoder takes about 3 seconds for 16 tiles, so extrapolating to the model's roughly 415,000 tiles gives about 25 hours **per configuration** before evaluation; a ROCm batch encoder is needed for the full eight-condition run. No sustained learned-neural-backward or true-task curvature optimizer comparison was added beyond the earlier small oracle and single-step checks. The inverse predictor is factorized bitwise rather than autoregressive, and 24-bit scaling was not run. The transformer used only one training shard and one seed. The image and text test data have already been exposed, so fresh confirmation remains necessary for any strong generalization claim.

## Reproduction

Run from the repository root in a context with `/dev/kfd` and `/dev/dri/render*` exposed. The pinned QTIP checkout must be available at `/tmp/qtip-official` for upstream block LDL and Hadamard source checks.

```sh
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m experiments.reference.qtip.tailbite
.venv/bin/python -c "from pathlib import Path; from experiments.reference.qtip.tailbite import run; run(out=Path('experiments/results/reference/qtip-gaussian-tailbite-normalized'), normalize=True, names=('3inst',))"
.venv/bin/python -m experiments.reference.qtip.pipeline_components
.venv/bin/python -m experiments.reference.qtip.smol_components
.venv/bin/python -m experiments.reference.qtip.submatrix_ablation
.venv/bin/python -m experiments.evaluation.int_controls --device cuda --dtype float32 --tokens 2048 --methods original int4_group64 int8_channel --out experiments/results/activation_ptq/int_controls_grouped
.venv/bin/python -m experiments.training.low_rate_qat --require-gpu --epochs 20 --methods scalar2 binary mul120_1bpw mul120_0.5bpw mul120_0.25bpw hash32_0.5bpw lowrank2 --out experiments/results/procedural_training/low_rate_20epoch
.venv/bin/python -m experiments.training.latent_qat --require-gpu --epochs 20 --width 32 --method discrete_jacobian --out experiments/results/procedural_training/latent_025_20epoch
.venv/bin/python -m experiments.training.inverse_scaling --require-gpu
.venv/bin/python -m experiments.training.tinystories_transformer --require-gpu --steps 1600 --eval-every 400 --methods dense scalar2 procedural2 procedural1 --out experiments/results/procedural_training/tinystories_transformer_1600
.venv/bin/python -m experiments.training.tinystories_transformer --require-gpu --steps 1600 --eval-every 400 --methods latent2 --out experiments/results/procedural_training/tinystories_latent2_1600
.venv/bin/python -m experiments.training.export_tinystories experiments/results/procedural_training/tinystories_transformer_1600
```

The TinyStories run requires the public first training shard and official validation shard in `data/tinystories/data/`, as named in `tinystories_transformer.py`. The five-epoch breadth and surrogate comparisons have their exact arguments in each result directory's `config.json`.
