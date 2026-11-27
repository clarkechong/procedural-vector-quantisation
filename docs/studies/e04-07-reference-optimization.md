# Experiments 04–07 execution report

## Scope and classification

This run completed reference *components* and bounded local adaptations. It did not reproduce published Llama-2 model quality. The checkpoint is SmolLM2-135M (`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`). Local test rows had been seen in earlier work; no result here is a blind final-test result. See `environment.json` for pinned code and versions, and the per-experiment manifests/configs for token or image splits.

The [QTIP source](https://github.com/Cornell-RelaxML/qtip) was pinned at `e90c6688c8dfae326a3a81b5eb032db7c6680ec0`; the [SmoothQuant source](https://github.com/mit-han-lab/smoothquant) at `c61476d728e42ae0d8a35e7e78494edcac3237b5`. Both remained in separate `/tmp` checkouts. This session has no Hugging Face token, and an explicitly unauthenticated request for the gated Llama-2-7B repository returned 401. That does not establish whether the user's account has access. QTIP's released HYB configuration is visible, but upstream model import requires CUDA `qtip_kernels` and missing dependencies. The latter cannot be called a reproduced MI300X model result.

## Reference and numerical gates

- **R0, component-validated:** Identity serialization/reload matched all 273 model state tensors, logits exactly, and NLL exactly on 127 next-token targets. Existing open-trellis tests and new weighted Viterbi tests match exhaustive optima, including zero and strongly unequal weights. A small exhaustive tail-biting fixture checks the boundary constraint. The evaluator test checks the single label shift and context reset.
- **R1, component-validated:** The independent channel-scale implementation matches upstream on shared-input projections in FP64. On SmolLM2, applying upstream smoothing to all 30 layers changed FP32 logits by at most `1.3828e-4` and NLL by `-4.77e-7` on a separate 128-token validation slice. Calibration used 128 local tokens, not upstream's 512 Pile sentences. The official Llama-2 W8A8 PPL target was not tested.
- **R2.1, component-validated:** Independent NumPy 1MAD and 3INST tables match all 65,536 upstream states exactly. A 256-vector, length-256, 2-bpw *open-boundary* standard-Gaussian pilot produced MSE `0.06566` (1MAD, 95% sampling interval `[0.06481, 0.06651]`) and `0.07087` (3INST, `[0.07029, 0.07144]`). The paper Table 1 reference uses its own boundary/scale conventions; these numbers are not a reproduction of its `0.069` targets.

## Experiment 04: functional PTQ

One SmolLM2 layer-15 down projection was quantized with fixed `L=5, k=2` and 120-byte map. Calibration used validation token offsets 0–511; evaluation used offsets 512–1023 (508 valid targets at context 128). A/B/C/D used identical codebook, row layout and scale-fitting budget; D used transformed activation second moments `h/S²`. The alpha-one B/D assignments and metrics agree to numerical tolerance, and alpha-zero B=A and D=C. Values in `activation_ptq/pilot/metrics.csv` show that local activation NMSE and NLL rankings differ. The best validation NLL change was `-0.00958` nats/token (B, alpha -0.5); no reliable functional benefit follows from a single projection and 508 targets. Original-space weight NMSE for that row was `0.1415`, activation NMSE `0.2237`. Transform vectors and byte-aligned row payload are included in `stored_bpw`; B/D cost about `2.061` bpw for this matrix, versus about `2.005` for A/C.

For all 210 projections, the matched FP32 127-target smoke test gave dense PPL `22.4615`, two-bit scalar `2,352,128`, and prior `L5,k2` window `7,629,407`. Scalar used symmetric uniform levels `[-3,-1,1,3]`, one fitted FP32 scale per matrix, and byte-aligned row packing. Its stored rate was `2.000063` projection bpw, and weight NMSE `0.14284`. The earlier window result had `2.01178` stored projection bpw. All are unusable model-quality outcomes; the short smoke sample is a diagnostic, not a stable final PPL estimate. A BF16 scalar smoke run is retained separately and is not mixed into the FP32 comparison.

## Experiments 05–07: optimisation

The tiny two-tile linear teacher/student oracle evaluates all 65,536 paired codeword states under the *true full-batch task loss*, including cross-tile interactions. Its CSV retains planted and off-codebook cases for discovered, scalar and random codebooks. This is a small representability/optimisation fixture, not an LLM result.

MNIST and Fashion-MNIST used official image/label files; the 60,000 training examples were split into 55,000 train and stratified 5,000 validation, with the official 10,000 test examples evaluated once at epoch 20. Three paired initialisation seeds (11, 17, 23), 20 epochs, batch 256 and identical data order were used. Each four-weight/8-bit codebook gave 2-bit projection weights; including 394 FP32 biases and three FP32 scales gives `482,208` deployed bits or `2.05412` bits per linear weight. Dense shadow weights and Adam state were training resources. Export decoder parity was exact, and all 256 codewords were used.

| Dataset | Dense test accuracy | Scalar shadow-QAT | Fixed procedural shadow-QAT |
|---|---:|---:|---:|
| MNIST | 97.91% ± 0.15% | 94.76% ± 0.04% | 95.08% ± 0.17% |
| Fashion-MNIST | 88.95% ± 0.32% | 85.21% ± 0.14% | 85.72% ± 0.30% |

SD is across three seeds, not a document bootstrap CI. The procedural method is trainable by dense-shadow STE and meets the proposed within-one-point scalar criterion on both datasets. It is 2.83 and 3.23 percentage points below dense, respectively. No claim is made for LLM trainability. Direct projected SGD with joint learning rate 0.1 changed zero codewords per epoch and reached 17.43% procedural MNIST mean test accuracy; biases accounted for its limited progress. A one-batch sweep found learning rate 10 changed about 7.7% of procedural tiles and lowered that batch's loss, but three-seed training at joint rate 10 and at weight rate 10/bias rate 0.1 remained unstable and near chance. All failed trials are retained.

The eight-bit finite-difference Jacobian matches independent bit forcing and zero/nonzero influence, and exact Bernoulli expectation gradients match finite differences. These are surrogate *component* checks; no surrogate was used as a backward pass during network training. The learned target-only inverse trained on 50,000 synthetic targets, validated on 5,000, and tested on 10,000 independent targets. Top-4 proposals plus current seed reached the exact nearest codeword on 99.93%, and top-16 on 100%; random-16 plus current reached 42.97%, one-bit neighborhood plus current 43.11%. Training took 2.32 seconds on MI300X. Exact CPU lookup of 10,000 targets took 0.077 seconds, so the neural inverse has no established cost advantage over a 256-entry exact codebook.

On 7,500 local targets from 25 batches of an independent MNIST shadow-QAT training run, 3,256 targets had a better seed than the current one. Top-4 learned proposals plus current matched the exact nearest seed on all 7,500; one-bit and random-16 proposals matched on 58.93% and 59.21% overall, mostly because the current seed was already optimal. Their hit rates restricted to nontrivial targets were 5.41% and 6.05%. A finite-difference linear proposal using the exact one-bit Jacobian hit 4.88% of nontrivial targets and captured 22.1% of their attainable local improvement; the true one-bit-neighborhood search captured 23.9%. A separate one-batch, full-network update changed 4,501 of 58,688 tiles under exact projection; training/held-out batch CE fell from `2.3023/2.3048` to `2.1964/2.2515`. Learned top-4 produced nearly identical CE, with a few different seed assignments; the one-bit update achieved `2.2635/2.2803`. These are single-step findings, not a sustained optimiser or test-accuracy result.

## Decision and smallest next experiment

Official Llama-2 reference quality is still unsupported; no rescaling improvement in model quality is established. Shadow-STE makes this fixed representation trainable on two small image tasks, while the tested direct projection settings fail. The learned inverse solves local nearest-codeword targets on synthetic and held-out task-gradient traces, but has no demonstrated sustained training or runtime advantage. The smallest experiment that distinguishes a missing QTIP pipeline from poor fixed-map geometry is to port one pinned official QTIP 1MAD projection to SmolLM2, verify Hessian, RHT, BlockLDLQ and export intermediates against upstream, then swap only the codebook under an identical trellis layout and effective rate.

## Reproduction commands

From the repository root, with the pinned upstream checkouts at `/tmp/qtip-official` and `/tmp/smoothquant-official`, run:

```sh
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m experiments.reference.qtip.gaussian_pilot
.venv/bin/python -m experiments.evaluation.identity_codec_check
.venv/bin/python -m experiments.reference.smoothquant.model_parity
.venv/bin/python -m experiments.evaluation.activation_ptq_pilot
.venv/bin/python -m experiments.evaluation.scalar_model --device cuda --dtype float32 --tokens 128 --context 128 --out results/e04-07-reference-optimization/activation-ptq/scalar-model-float32
.venv/bin/python -m experiments.training.tiny_oracle
.venv/bin/python -m experiments.training.mnist_qat
.venv/bin/python -m experiments.training.mnist_direct
.venv/bin/python -m experiments.training.direct_step_sweep
.venv/bin/python -m experiments.training.learned_inverse
.venv/bin/python -m experiments.training.real_trace_inverse
.venv/bin/python -m experiments.training.task_update_check
```

GPU commands require an execution context with `/dev/kfd` and `/dev/dri/render*` exposed. The separate source clones were obtained using `git clone --depth 1` at the commits recorded in `environment.json`.
