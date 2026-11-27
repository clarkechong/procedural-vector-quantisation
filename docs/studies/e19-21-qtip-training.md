# Experiments 19–21: narrowed research programme, first execution gate

27 November 2026. Interim result. The primary decision question is whether a computational weight representation improves the joint quality, stored-size, and inference-cost frontier against scalar and computational references.

## Experiment 19: QTIP encoder and map substitution

The new batched MI300X trellis implementation follows the pinned QTIP two-pass midpoint overlap rule for lookup-free V=1, L=16, K=2 maps. It stores predecessor choices as uint8 rather than a dense transition matrix. On 256-value tiles, all checked packed state paths matched the independent CPU implementation exactly: 8/8 1MAD at batch 8, 2/2 1MAD at batch 64, and 2/2 3INST at batch 64. The 64-tile runs took 0.202 s and 0.200 s inside the GPU encoding call, respectively, with 365 MB peak allocation. These are first-call throughput figures, not sustained whole-model throughput or direct parity with the upstream CUDA kernel.

A complete 576×576 SmolLM2 layer-15 `q_proj` was encoded with the same 16×16 tile order, 2-bpw trellis payload, 512 dense-input calibration tokens, 512 disjoint held-out activation tokens, and block-LDL feedback. The binary file is 82,948 bytes: 82,944 trellis bytes plus one FP32 scale. A separate decoder reads the serialized bytes in deterministic tile order and reconstructs exactly the matrix returned by the encoder. Results are for a **no-RHT ablation**, so they do not reproduce the native QTIP pipeline or model perplexity.

| Map and feedback | Weight NMSE | Held-out layer-output NMSE | Payload bpw |
|---|---:|---:|---:|
| Normalized 1MAD, no LDL feedback | 0.09834 | 0.10000 | 2.000 |
| Normalized 1MAD, BlockLDLQ | 0.21266 | **0.03924** | 2.000 |
| Normalized multiplier-120, BlockLDLQ | 0.31496 | 0.12222 | 2.000 |

The full-projection comparison reinforces the prior crop result: the fixed multiplier-120 map is substantially worse than 1MAD for post-training reconstruction at the same trellis rate. The LDL feedback trades larger ordinary weight error for much lower activation-weighted output error. It does not establish downstream language-model quality. The local block-LDL factorization was previously checked against pinned upstream component math; the **full projection feedback, RHT signs, and swizzled upstream payload still need direct upstream parity**. Before whole-model claims, that parity, native regularization and scale settings, sequential calibration, 3INST full projection, and model evaluation remain required.

## Experiment 20: actual training-map audit

The training code calls `make_book("mul120", width)`, which uses a four-coordinate shift cycle with an XOR offset on each later group. This differs from the literal eight-bit shift extension audited in Experiment 13–18. At widths 4, 8, 16, and 32, corresponding to 2, 1, 0.5, and 0.25 payload bpw, all 256 codewords are distinct; every coordinate varies and has nonzero RMS; the coordinate-function counts are 4, 8, 16, and 32. Thus the earlier 2/1-bpw TinyStories result cannot be attributed to literal structurally zero coordinates. This audit does not prove that the representation is more expressive than trained VQ or structured-sparsity controls.

Three paired seeds (11, 17, 23) trained the same 4-layer, 128-hidden byte-level transformer for 6,400 updates on the available first TinyStories training shard. Within a seed, all scratch arms started from identical tensors and used identical sampled sequence IDs. Quantization scales were calculated from those loaded weights. Scalar INT4, INT2, binary, multiplier-120 at 2/1/0.5 bpw, and dense were trained from scratch. The dense-to-QAT controls copied the dense checkpoint after 4,800 updates and continued 1,600 updates with a fresh AdamW optimizer on the same remaining sequence IDs. Selection and confirmation examples are distinct story IDs from the public validation shard; the latter were not used for checkpoint choice here, but this public dataset and prior TinyStories work are exposed, so confirmation is **not blind**. All arms were evaluated at the fixed 6,400-update budget; convergence has not been established.

| Arm | Mean confirmation CE, 3 seeds | Total stored bits | Projection payload bpw |
|---|---:|---:|---:|
| Dense scratch | 1.15760 | 28,532,736 | 32 |
| Scalar INT4 scratch | 1.29187 | 6,513,152 | 4 |
| Scalar INT2 scratch | 1.39517 | 4,940,288 | 2 |
| Procedural 2 scratch | 1.38716 | 4,940,288 | 2 |
| Binary scratch | 1.59017 | 4,153,856 | 1 |
| Procedural 1 scratch | 1.57995 | 4,153,856 | 1 |
| Procedural 0.5 scratch | 1.74211 | 3,760,640 | 0.5 |
| Dense→scalar INT2 QAT | 1.28172 | 4,940,288 | 2 |
| Dense→procedural 2 QAT | 1.28449 | 4,940,288 | 2 |
| Dense→binary QAT | 1.43988 | 4,153,856 | 1 |
| Dense→procedural 1 QAT | 1.51979 | 4,153,856 | 1 |

Procedural-minus-scalar paired confirmation differences at 2 bpw are -0.01532, -0.01221, and +0.00350 nats/token; mean -0.00801. Procedural-minus-binary at 1 bpw are -0.01198, -0.01988, and +0.00119; mean -0.01022. These are small, inconsistent per-seed scratch advantages. After dense-to-QAT continuation, procedural minus scalar2 is +0.00276 on average, and procedural minus binary is +0.07992. The data support a limited scratch-training advantage under this exact setup, not a robust compression frontier improvement. The smaller-dense, trained-VQ, native-QTIP, structured-sparsity, and multi-shard controls remain outstanding. Stored bits include the dense non-projection tensors and per-projection scales but assume the fixed procedural program is shared; no actual packed TinyStories binary was exported in this run. See `tinystories_frontier/summary.csv`, `curves.csv`, and `config.json`.

## Experiment 21: hardware status

The MI300X encoder is now usable at full-projection scale: 1,296 tiles of the 1MAD BlockLDLQ projection took 1.96 s in the encoding/feedback stage, roughly 662 tiles/s. Calibration/model loading and payload decoding add wall time. This is **encoder** throughput, not fused inference performance. The requested batch-1/4/16/64 decode-and-multiply kernel comparison and HBM, VGPR, occupancy, and MFMA measurements remain open.

## Decision gates

The present evidence supports continued implementation of the QTIP reference and a stronger matched language-model frontier, but it does **not** yet pass the representation-quality, language-model-training, hardware-latency, or generator-synthesis continuation criteria in the proposed plan. The first decisive next gate is native full-projection RHT/BlockLDLQ parity against pinned upstream and then whole-model SmolLM2 evaluation before Llama-2 confirmation. TinyStories now has a paired 6,400-step exploratory comparison, but convergence, training-shard breadth, trained VQ, smaller-dense, and structured-sparsity controls remain. Measured fused inference is also outstanding.

## Reproduction

```sh
.venv/bin/python -m experiments.reference.qtip.batched_trellis --batch 64 --check 2 --require-gpu --out results/e19-21-qtip-training/qtip_batched_1mad_64.json
.venv/bin/python -m experiments.reference.qtip.batched_trellis --map 3inst --batch 64 --check 2 --require-gpu --out results/e19-21-qtip-training/qtip_batched_3inst_64.json
.venv/bin/python -m experiments.reference.qtip.full_projection --map 1mad --normalize --ldlq --require-gpu --out results/e19-21-qtip-training/qtip_full_qproj_1mad_normalized.json
.venv/bin/python -m experiments.reference.qtip.full_projection --map 1mad --normalize --require-gpu --out results/e19-21-qtip-training/qtip_full_qproj_1mad_no_ldlq.json
.venv/bin/python -m experiments.reference.qtip.full_projection --map mul120 --normalize --ldlq --require-gpu --out results/e19-21-qtip-training/qtip_full_qproj_mul120_normalized.json
.venv/bin/python -m experiments.training.tinystories_map_audit
.venv/bin/python -m experiments.training.tinystories_frontier --require-gpu
.venv/bin/python -m unittest discover -s tests
```

The tests passed: 50 run, 1 skipped. GPU commands ran outside the workspace sandbox to access `/dev/kfd`.
