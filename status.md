# Experiment status — 27 November 2026

**Experiments 13–18 interim update:** See the [execution report](experiments/results/experiments-13-18/report.md). SeedLM 3/4-bpw GPU all-seed pilot matched the exhaustive CPU scorer on 128 blocks each, but is not model-scale fast. The historical matrix-wide INT8 failure was localized to activation-sensitive projections. Three paired MLP seeds completed scratch/FT, legal alignment, release and independent-codec transfer; no different disconnected basin or transferable low-rate advantage is established. The full reference encoders, TinyStories extension, QTIP A–H, and fused MI300X inference remain open.

**Current Llama-2 reference update:** access is now available. The released
QTIP HYB checkpoint reproduces 5.8608 WikiText-2 PPL against the paper's
5.86; its dense control gives 5.1164 against 5.12. SmoothQuant's released
alpha-0.85 scales and upstream fake W8A8 path also run. See the
[Llama-2 follow-up](experiments/results/llama2-reference-followup/report.md).
The historical blocked entries below describe the earlier 04–07 run.

| Gate | Status | Evidence and next dependency |
|---|---|---|
| R0 identity/evaluator | component-validated | 273 state tensors serialize/reload exactly; FP32 logits and 127-target NLL match bit for bit after BF16 reload. Exact open weighted-trellis and small circular boundary oracles pass. |
| R1 SmoothQuant | component-validated; full reference blocked in this session | Pinned upstream channel transform matches independent FP64 implementation; upstream full-network transform on SmolLM2 changes max FP32 logit by 1.38e-4. This session has no Hugging Face token; an explicitly unauthenticated Llama-2-7B request returned 401. This does not establish whether the user's account has access. Released activation scales are absent locally. |
| R2.1 QTIP | component-validated | All 65,536 1MAD and 3INST states match pinned upstream functions exactly. A 256-vector open-boundary Gaussian pilot gave MSE 0.06566 and 0.07087 with sampling intervals; this does not reproduce the paper's tail-biting Table 1 setup. |
| R2.2–3 QTIP full reference | blocked | Released HYB model config is public, but upstream import needs CUDA `qtip_kernels` and `glog`; the matching Llama-2 tokenizer/model is gated. No full quantizer/evaluator reproduction or paper-quality claim. |
| 04 functional PTQ | adaptation-only | One-projection 508-target scaling/weighting pilot; FP32 full-model 127-target scalar/window comparison. All model-wide 2-bpw PPL values are catastrophic. No fresh blind final test or matched full QTIP adaptation. |
| 05 trainability | adaptation-only | Three seeds × 20 epochs on MNIST and Fashion-MNIST with dense, scalar shadow QAT and procedural shadow QAT. Direct projected SGD controls and step-size sweep are retained. |
| 06 surrogate | component-validated | Eight-bit exact finite-difference Jacobian and exact Bernoulli expectation gradient verified; no real-task surrogate training comparison yet. |
| 07 learned inverse | exploratory | 50k/5k/10k synthetic inverse pilot, 7,500 held-out MNIST gradient-trace targets, and one full-network task-loss update. Top-4 proposals matched the local oracle on the trace, but no sustained learned-optimizer training or cost advantage is established. |
| Hardware | exploratory prior work | Experiment 03 isolated MI300X kernel exists; no new matched INT2/INT4/GEMM throughput claim. |

The reference and detailed metrics are in [the report](experiments/results/experiments-04-07/report.md). The next controlled comparison is a pinned QTIP 1MAD one-projection SmolLM2 port with Hessian/RHT/BlockLDLQ intermediate parity, then a fixed-map swap under the same layout and rate.

## Experiments 08–12 update — 27 November 2026

Detailed measurements and limits are in [the 08–12 report](experiments/results/experiments-08-12/report.md).

| Experiment | Current evidence | Remaining gate |
|---|---|---|
| 08 QTIP and controls | Independent tail-biting Gaussian 1MAD/normalised 3INST reproduce the paper's rounded 0.069 MSE. Synthetic 64D block LDL factors match pinned upstream exactly; actual SmolLM2 layer-15 1,536D activation Hessian/RHT/block LDL checks pass. A matched 16×256 local A–H ablation favors QTIP 1MAD with RHT+compensation; all eight circular payloads decode exactly. INT8 channel and INT4 group-64 controls were evaluated on 2,032 targets. | Full-projection BlockLDLQ parity, model-wide compressed QTIP export, eight model-wide A–H ablations, and complete upstream intermediate parity. |
| 09 low-rate QAT | Three seeds, five-epoch broad sweep and 20-epoch critical-rate sweep on MNIST/Fashion-MNIST. Procedural 1-bpw beats binary at the same stored rate, but not scalar 2-bpw. At 0.5 bpw the tested generators fall further behind. | Fresh blind test and broader architecture/low-rank search. |
| 10 surrogates | Full-network five-epoch exact discrete-Jacobian and Bernoulli latent training succeed; degree-two Boolean surrogate fails. 0.25-bpw discrete-Jacobian runs extend to 20 epochs with 3.98× fewer continuous parameters than shadow QAT, but lower accuracy. | Learned-neural backward and sustained curvature-aware true-task optimisation comparison; measured total peak training memory. |
| 11 inverse scaling | 12/16/20-bit synthetic bit-predictor tests with exhaustive oracles and explicit random codebook control. Simple bit extraction inverts, mixed/hash maps do not under the tested budget. | Autoregressive or iterative inverse, 24-bit selected oracle, real task-gradient scaling, useful cost advantage. |
| 12 TinyStories | One-seed 4-layer 128-hidden byte-level transformer trained on a public TinyStories shard. Dense, scalar 2-bit, procedural 2/1-bit scratch, PTQ, fine-tuning, and discrete-Jacobian latent training; independent packed export and decoder parity. | More seeds, converged training/full corpus, fresh text test. |
