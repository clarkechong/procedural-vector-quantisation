# Llama-2-7B reference follow-up — 27 November 2026

Hugging Face access to `meta-llama/Llama-2-7b-hf` now works. This follow-up uses
base revision `01c7f73d771dfac7d292323805ebc428287df4f9`, the pinned
WikiText-2 raw Parquet revision `b08601e04326c79dfdd32d625aee71d232d685c3`,
PyTorch 2.9.1+rocm6.4, and an MI300X. The base checkpoint and both released
reference artifacts were downloaded into ignored `data/` directories.

## Published references that now run

| Reference and protocol | Published PPL | Local PPL | Local implementation |
|---|---:|---:|---|
| SmoothQuant dense, 166 × 2048, BF16 | 5.474 | 5.4733 | Local evaluator, one BOS |
| SmoothQuant W8A8, no smoothing | — | 5.5838 | Pinned fake quantizer, one BOS |
| SmoothQuant W8A8, alpha 0.85 | 5.515 | **5.5132** | Pinned smoothing and fake quantizer, released activation scales, one BOS |
| QTIP dense, 83 × 4096, FP16 | 5.12 | 5.1164 | Local evaluator, one BOS |
| QTIP released HYB 2-bit checkpoint | 5.86 | **5.8608** | Pinned PyTorch decompressor, local dense forward adapter |

SmoothQuant uses upstream commit `c61476d728e42ae0d8a35e7e78494edcac3237b5`
and activation-scale revision `5a18a8761e45587e1c06cee58a33c1c1e51f5d4b`.
The source transforms and fake quantizer are imported unchanged. Its upstream
`ppl_eval.py` supplies the windowing, tokenizer, dtype, and W8A8 settings; the
local evaluator reads the pinned Parquet file and counts the 2047 scored targets
in each complete 2048-token window. The upstream script multiplies each
window's mean loss by 2048 and divides by 2048, which gives the same average
over complete windows. The BOS behavior was checked against `AutoTokenizer`.
These W8A8 measurements are fake quantization quality, not INT8 throughput.
Smoothing improved the unsmoothed W8A8 control by 0.0706 PPL and stayed
0.0399 PPL above the matched BF16 dense run.

QTIP uses released HYB model revision `392a351a81d19c2af0ab0da9203e641cd8e441f9`
and pinned source commit `e90c6688c8dfae326a3a81b5eb032db7c6680ec0`.
The released trellis payload is swizzled for the native kernel. A natural
`unpack_trellis` interpretation gave weight NMSE 2.18 on layer-0 `q_proj` and
was rejected. Extracting the pinned `decode_compressed` PyTorch body and
`quantlut_sym` produced a plausible 0.1017 NMSE and 0.951 correlation on that
projection; the full checkpoint reproduced the paper's 5.86 PPL. The 32-layer
run loads non-projection tensors from the released checkpoint and replaces
all 224 quantized projections with equivalent dense forward modules using
the checkpoint's `SU`, `SV`, trellis and learned lookup table. The Hadamard
source is extracted from the pinned repository; its cleanup call is a no-op.
This verifies released-model quality on ROCm. It does not benchmark the native
CUDA kernels or reproduce the no-fine-tuning 1MAD/3INST encoder.

The released QTIP SafeTensors file is 2,149,381,744 bytes. Its projection
payload is 2.00674 bpw over 6,476,005,376 projection weights when all trellis,
`SU`, `SV`, and lookup table bytes are counted. The file also contains dense
embeddings, output head and norms. The corresponding two base SafeTensors
shards total 13,476,876,272 bytes. These are file sizes, not resident GPU
memory or decoding throughput.

## Additional controls and SeedLM gate

At the SeedLM 2048-token protocol, dense FP16 Llama-2 scored 5.4721 PPL. A
simple per-row group-64 INT4 control scored **5.7311 PPL at 4.25 projection
bpw** and 0.01189 projection-weight NMSE. This is ordinary uniform quantization,
not GPTQ or the published SeedLM 4-bit encoder.

The exact 3- and 4-bpw SeedLM encoder was audited on 1,344 and 2,016 Llama
weight blocks. The 1,024-seed indexed shortlist had NMSE 0.11046 / 0.03911;
exhaustive 65,535-seed search improved these to 0.09349 / 0.03533.
Projection-bound screening returned exactly the exhaustive fields for both
audits. A full 4-bpw CPU pass was started and stopped after three projections:
it was advancing too slowly for an interactive model-wide run. The GPU
projection-bound pilot is not yet a completed encoder. Consequently there is
**no local full-model SeedLM 3- or 4-bpw perplexity** in this follow-up, and
the partial payload log is not a model-quality result. The paper reports
5.7 at 4 bpw and 6.6 at 3 bpw, under its own implementation.

The full no-fine-tuning QTIP 1MAD/3INST quantization run and the model-wide
SmolLM2 A–H codebook ablation from Experiment 08 remain separate work. Their
absence should not be conflated with the successfully evaluated released HYB
checkpoint.

## Reproduce

The checkpoints and Parquet files are in ignored `data/`; `/tmp/qtip-official`
and `/tmp/smoothquant-official` are pinned source checkouts. Run GPU commands
with access to `/dev/kfd` and `/dev/dri/render*`. `uv sync --locked` currently
reinstalls a CUDA-only PyTorch wheel in this environment, so verify a ROCm
build before running on MI300X.

```sh
.venv/bin/python -m experiments.reference.smoothquant.llama2_eval --method original
.venv/bin/python -m experiments.reference.smoothquant.llama2_eval --method w8a8
.venv/bin/python -m experiments.reference.smoothquant.llama2_eval --method smooth_w8a8
.venv/bin/python -m experiments.reference.qtip.released_llama
.venv/bin/python -m experiments.evaluation.generator_model --checkpoint data/llama-2-7b-hf --data data/wikitext-2-raw-v1 --out runs/llama2-7b/qtip-dense-control --methods original --device cuda --dtype float16 --threads 4 --context 4096 --windows 83 --batch 1 --bos --calibration-tokens 0
.venv/bin/python -m experiments.evaluation.seedlm_model --checkpoint data/llama-2-7b-hf --data data/wikitext-2-raw-v1 --out runs/llama2-7b/int4-group64 --methods int4_group64 --device cuda --dtype float16 --threads 8 --context 2048 --windows 166 --batch 1 --bos --calibration-tokens 0
```

Small raw JSON/CSV files are preserved alongside this report. The main
checkpoint weights stay in ignored `data/`, and the working run directories
stay in ignored `runs/`.
