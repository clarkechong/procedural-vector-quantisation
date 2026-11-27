# Experiments 13–18: execution progress and decision gates

27 November 2026. Repository `9043d85` plus the uncommitted working tree. This is an **interim execution report**. It does not claim completion of the full handoff.

## Executive findings

1. **Independent reference encoding:** The all-seed ROCm SeedLM pilot produced exactly the same packed fields as the native exhaustive scorer on 128 Llama-2 blocks at each of 4 and 3 bpw. It reached only 141 and 133 blocks/s, respectively, on this small pilot, so neither full-model SeedLM quality nor independent QTIP no-FT quality has been reproduced. The previously matched released HYB and SmoothQuant results remain released-artifact/reference evaluations, not new encodes.
2. **Activation importance:** The historical whole-matrix INT8 failure is a real activation-sensitivity failure, not a broken signed-byte decoder. At SmolLM2 layer 11 `down_proj`, the 50-step matrix-wide odd-level INT8 control has weight NMSE 0.004474 yet held-out dense-input output NMSE 0.7594 and raises NLL by 7.359. A zero-inclusive per-row INT8 control gives output NMSE 0.0000375 and ΔNLL 0.00205. Across a 21-projection predictor bank, activation-aware metrics have weak and inconsistent rank correlations with ΔNLL; their out-of-sample predictive advantage remains unestablished. A one-projection 2-bpw A/B/C/D pilot changes quality, but all candidate models are badly damaged and transformed variants incur input-scale metadata.
3. **Constrained training and alternative optima:** Three paired MNIST MLP seeds were run at both 12 and 20 epochs. At 20 epochs, procedural 2-bpw QAT validation CE is 0.1903±0.0074 versus scalar 2-bit CE 0.2082±0.0035 and dense 0.1129±0.0052. Dense continuation from the projected initial weights reaches CE 0.1124, showing the initial projection itself does not prevent dense learning. Legal hidden-unit alignment removes the large raw interpolation barrier between independently initialized dense models. Procedural and paired dense endpoints have no positive aligned linear barrier in these runs, but procedural endpoints are materially worse. There is **no evidence here of a separate disconnected minimum or transferable intrinsic compressibility**. Independent-codec transfer is weaker for the procedural source than for the scalar source, and release does not preserve a general low-rate codec advantage.
4. **MI300X efficiency:** The SeedLM PyTorch implementation is a parity pilot, not a usable full encoder. Packed MLP endpoints are exactly 60,282 bytes per model, 2.000 projection bpw and 2.0543 bits per original weight including three FP32 scales, FP32 biases and a six-byte header. Independent reload changes logits by at most 8.6e-6 on 100 images. No new fused compressed Llama/SmolLM inference latency, QTIP encoder throughput, or measured training-memory improvement is established.

## Reference ledger

| Configuration | Published/reference | Local status | Classification |
|---|---|---|---|
| Llama-2 dense, SmoothQuant protocol | 5.474 full WikiText PPL | Prior full 5.4733; new two-window dense smoke 5.09764 on 4094 targets | reference-reproduced for dense quality; short smoke is not the full score |
| Llama-2 SmoothQuant W8A8 alpha .85 | 5.515 | Prior full 5.5132. New two-window rerun stopped during slow upstream fake-quant setup, before scoring | reference-reproduced previously, no new result |
| QTIP released HYB 2-bit | 5.86 | Prior full 5.8608; pinned dense forward adapter | released-artifact-validated |
| SeedLM native 4/3 bpw | 5.7/6.6 per supplied handoff | 128-block all-seed field parity each, no full-model checkpoint/PPL | component-validated only |
| QTIP lookup-free no-FT 4/2 bpw | 5.17/7.05 or 6.82 per supplied handoff | Existing Gaussian/crop component results; no full projection/model encode | blocked by missing fast encoder and full BlockLDLQ export parity |

SeedLM pilot details: 4-bpw 128/128 field agreement, max excess decoded SSE 0, 0.907 s GPU versus 0.0415 s native CPU, 97 MB peak allocated. A larger one-tile batch took 0.867 s and 1.267 GB peak. At 3 bpw, 128/128 field agreement, max excess SSE 0, 0.965 s GPU versus 0.0677 s CPU, 111 MB peak. These are small samples of contiguous layer-0 `q_proj` weights, not sustained whole-projection throughput or multi-layer parity. Neither a full 4-bpw nor a full 3-bpw SeedLM model was encoded. The GPU implementation is currently slower than the native CPU on this sample and requires a different batching/pruning design before a model-scale run is responsible.

The existing QTIP 16×256 crop regression was rerun: 1MAD NMSE 0.08520 without transforms and 0.06146 with RHT+LDL; multiplier-120 0.13780 and 0.10237. These reproduce the earlier crop only. The attempted default tail-biting rerun was interrupted before reporting; earlier artifacts were retained.

## INT8 control and activation hypothesis

The historical one-scale odd-level INT8 full-model run gave PPL 11.97 million and projection weight NMSE 0.02165 on 2032 targets. Increasing the same scale fit to 50 iterations lowers NMSE to 0.002762 but PPL remains 8.08 million. Independent FP32 dequantization agrees with the encoder. Odd versus zero-inclusive levels make little difference at the same matrix granularity/fit; row/channel granularity changes the outcome substantially. For five representative matrices, 50-step fitting reduces odd-level NMSE from roughly 0.021–0.076 to 0.0015–0.0158. The exact cause is concentration of error in activated channels and sensitive projections, with layer-11 `down_proj` the largest isolated ΔNLL on the 256-token diagnostic. Thus the old catastrophic run is **not** an acceptable generic INT8 quality control. The previous per-channel INT8 full-model PPL 36.37 is the relevant positive control, with its own metadata rate.

The 21-projection bank used fixed dense inputs from 2048 validation tokens and evaluated one replacement at a time on 2032 exposed test targets. For the 50-step matrix INT8 family, Spearman ρ between ΔNLL and plain/diagonal/full-second-moment NMSE is 0.123/0.244/0.271 across 21 projections; none is statistically persuasive, and no untouched candidate-family confirmation was performed. Within each projection across five disparate codecs, mean Spearman was 0.529 for plain NMSE, 0.514 diagonal, and 0.510 full; this comparison mixes rates and is descriptive only. The layer-11 example demonstrates that plain weight NMSE can hide severe output damage, but it does not establish a reliable global NLL predictor. Full second moment is a linear output-error metric, not a task-loss Hessian.

The A/B/C/D pilot used the entire layer-11 `down_proj` and two 256-codeword four-weight maps: independent scalar2 patterns and fixed multiplier-120. Each uses exactly 2 payload bpw; A/C add one FP32 matrix scale (2.00004 projection bpw), while B/D also store 1536 FP32 input scales (2.05559 projection bpw). At fixed global scale, assignment minimizes its declared diagonal additive SSE; three alternations jointly fit the scale and assignments. Dense validation inputs supplied RMS importance; single-projection test replacement supplied ΔNLL. For multiplier-120, A ΔNLL was 11.43, C 5.40, and best B/D in the predeclared alpha grid were 4.88/4.93 at alpha -0.5. Scalar2 A/C were 12.06/12.91; best B was 9.94 at alpha 0.75. These are **exploratory damaged-model results**, not successful compression or a positive-alpha finding. Both maps remain far worse than usable full-model controls. The B/D representation was evaluated through its mathematically equivalent original-domain matrix; explicit input scaling runtime and legal folding were not measured.

At alpha 0, A/B and C/D have identical code indices and decoded weights. At alpha 1, B/D cease to match in this real projection because two input channels hit the documented [1/16,16] scale clamp; the unclamped algebra identity passes a separate numerical test. The handoff's equivalence applies only without such clamping/floor changes.

## Matched training and geometry

All scratch arms share exact initial tensors and minibatch order by paired seed 11/17/23. Architecture 784–256–128–10, Adam 0.001, batch 256, 12 epochs. Scales are frozen layer RMS at the paired dense initialization. The old 5k MNIST training holdout is used as **exposed exploratory validation**; official test images were not used for selection or this report. Each epoch saved decoded weights, shadow tensors and optimizer state in ignored `runs/`. Dense→QAT starts at epoch 9 with a fresh optimizer and gets the same remaining data order as dense reset; release starts from actual decoded forward weights and uses a fresh dense optimizer for three epochs.

| Arm | Mean final validation CE | SD across 3 seeds |
|---|---:|---:|
| Dense scratch | 0.09510 | 0.00831 |
| Dense from procedural-projected initialization | 0.09541 | 0.00597 |
| Scalar2 scratch | 0.21840 | 0.00485 |
| Multiplier-120 2-bpw scratch | 0.19907 | 0.00639 |
| Dense epoch-9→scalar2 QAT | 0.22908 | 0.00134 |
| Dense epoch-9→multiplier-120 QAT | 0.21185 | 0.00143 |
| Scalar2 scratch→dense release, 3 epochs | 0.11360 | 0.00754 |
| Procedural scratch→dense release, 3 epochs | 0.10911 | 0.00537 |

Procedural minus scalar scratch CE is -0.01933 on average at identical packed storage. This is a finite-budget paired advantage, with only three seeds and a previously exposed validation split. The constrained endpoints remain far worse than dense. After legal ReLU hidden-unit permutation alignment (logits preserved within 1e-5), procedural endpoints still have mean normalized weight distance 0.762 from paired dense, versus 0.777 for scalar. Both have zero positive aligned linear interpolation barrier relative to endpoint interpolation on the 21-point grid. Independent dense seeds had raw barriers 0.41–0.48 CE, but legal alignment reduced them to 0–0.0085. Parameter distance and raw barriers alone therefore do not demonstrate different basins.

Independent-codec PTQ transfer on the same endpoints gives mean ΔCE: procedural source→scalar2 +0.0733; scalar2 source→procedural +0.0343; procedural source→scalar4 +0.00336; scalar source→scalar4 -0.01263. Each source's native codec is the expected zero-damage positive control. After three dense release epochs, procedural descendants incur +0.2639 under scalar2 and +0.1605 under their original procedural codec, so the released low-rate representability has not been retained. Absolute task quality, source CE and packed rate are in the raw cross-codec CSV. This matrix does not show an off-diagonal advantage transferable from procedural training. Alignment is diagnostic in dense space and is not claimed to preserve compressed tile membership.

The actual packed format stores a six-byte header, three FP32 scales, 394 FP32 biases, and 58,688 bytes of 2-bit payload: 60,282 bytes/model. The shared decoder program is identified by the format ID but its implementation bytes are not charged to a one-model file; any fleet-level rate comparison must include shared program cost. Independent reload changed weights by at most 7.5e-9 and logits by at most 8.6e-6 on 100 inputs. Dense FP32 storage for the same architecture is 940,584 bytes. Training uses FP32 shadow tensors and Adam state, so deployment compression does not imply training memory savings.

The 20-epoch extension is preserved separately in `landscape_mlp_20/` with the same three seeds, model, scale policy and data order. At its final epoch, procedural/scalar/dense CE means (SD) were 0.19026 (0.00744), 0.20823 (0.00354), and 0.11294 (0.00515). The paired procedural advantage is -0.01797 CE and persists with this longer budget. The dense validation curve worsens after approximately epoch 9 while both constrained curves still improve, so this is not a convergence-matched optimum comparison. Procedural→scalar2 independent-codec damage is +0.08047 CE, versus scalar2→procedural +0.03463; after five release epochs, the procedural descendant incurs +0.35238 under scalar2 and +0.22720 under its original map. All three procedural↔dense aligned 21-point paths still have zero positive endpoint-adjusted barrier. Independent dense controls have raw barriers 0.297–0.371, reduced to 0–0.00498 by legal alignment. The 20-epoch packed files are also 60,282 bytes each and reload within 5.3e-6 logits on 100 inputs. These outcomes reinforce the 12-epoch qualitative conclusion but remain exploratory on the exposed validation split.

The exact four-output multiplier-120 map is `int8(120*(s>>i))`. A literal eight-bit seed extended to width 16 has eight structurally zero coordinates; at width 32 it has 24. Cyclic shifts repeat eight coordinate functions. A zero-padded fixed eight-bit seed with five-bit, stride-two windows has 12/16 and 28/32 zero coordinates; **this is not the open-window stream with new payload bits**, whose width/rate are different. The enumeration is in `fixed_seed_audit.csv`.

## Efficiency, status and next gate

ROCm PyTorch 2.9.1+rocm6.4 accessed the MI300X outside the workspace sandbox; the in-sandbox device check remains false because `/dev/kfd` is hidden there. Full SeedLM encoding, QTIP GPU DP/BlockLDLQ export, SmolLM2 QTIP A–H, Llama confirmation, TinyStories 6400-step multi-seed training, sustained Experiment 16 optimizer runs, conditional Experiment 15 synthesis, and Experiment 18F native fused inference are **not completed**. These are substantive parts of the handoff. No negative full-model representation conclusion can be drawn from this partial phase.

The next highest-value gate is an exact accelerated QTIP 1MAD full-projection encoder with intermediate BlockLDLQ parity and independent export/decode; without that, the causal A–H and no-FT Llama comparisons cannot be interpreted. In parallel, an improved SeedLM GPU search needs a measured sustained projection throughput and parity across projection types before a full encode. For the alternative-optima question, extend paired training to a converged budget and fresh split, then test independent-codec transfer after release at matched source task quality. The current finite-budget result supports functional adaptation under the procedural constraint, not a new disconnected basin or intrinsic transferable compressibility.

## Reproduction and raw artifacts

All commands below exited 0 unless noted. GPU commands require host device access outside the workspace sandbox.

```sh
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m experiments.evaluation.int_control_diagnosis
.venv/bin/python -m experiments.evaluation.int_controls --device cuda --dtype float32 --tokens 2048 --methods original int8_fitted --out experiments/results/experiments-13-18/int8_fitted
.venv/bin/python -m experiments.evaluation.int_control_layer_ablation
.venv/bin/python -m experiments.synthesis.fixed_seed_audit
.venv/bin/python -m experiments.encoding.gpu_paper_seedlm --bits 4 --blocks 128
.venv/bin/python -m experiments.encoding.gpu_paper_seedlm --bits 3 --blocks 128
.venv/bin/python -m experiments.training.landscape_mlp --seeds 11 17 23 --epochs 12
.venv/bin/python -m experiments.training.landscape_dense_control
.venv/bin/python -m experiments.training.pack_mlp
.venv/bin/python -m experiments.training.landscape_mlp --seeds 11 17 23 --epochs 20 --out experiments/results/experiments-13-18/landscape_mlp_20
.venv/bin/python -m experiments.training.landscape_dense_control --result experiments/results/experiments-13-18/landscape_mlp_20
.venv/bin/python -m experiments.training.pack_mlp --result experiments/results/experiments-13-18/landscape_mlp_20
.venv/bin/python -m experiments.evaluation.activation_metric_bank
.venv/bin/python -m experiments.evaluation.activation_metric_bank --layers 11 --out experiments/results/experiments-13-18/activation_metric_layer11
.venv/bin/python -m experiments.evaluation.factorial_ptq
MPLCONFIGDIR=/tmp/pvq-mpl .venv/bin/python -m experiments.training.plot_landscape_mlp
MPLCONFIGDIR=/tmp/pvq-mpl .venv/bin/python -m experiments.training.plot_landscape_mlp --result experiments/results/experiments-13-18/landscape_mlp_20
MPLCONFIGDIR=/tmp/pvq-mpl .venv/bin/python -m experiments.evaluation.plot_activation_bank
```

The two-window SmoothQuant dense smoke exited 0. The two-window smoothed fake-W8A8 smoke was interrupted during upstream fake-quant module construction after the prior full validated result was confirmed; it yielded no new score. The default QTIP tail-biting rerun was also interrupted before emitting a new result. `git diff --check` and Python compilation of new modules passed; unit tests: 48 run, 1 skipped.

Small raw outputs, configs and plots are in this directory. The paired landscape plot is `landscape_mlp/landscape_summary.png`; detailed CSVs are adjacent. Heavy checkpoints and six 60,282-byte packed files are in ignored `runs/experiments-13-18/landscape_mlp/`.
