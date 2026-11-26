# Generator and SeedLM perplexity on SmolLM2-135M

The evaluation replaces all seven attention and MLP projection matrices in every decoder layer. That covers 106,168,320 weights across 210 matrices. The rest of the checkpoint stays at its original precision. The reference Transformers model loads weights decoded from the packed payloads and runs in FP32. Storage rates describe the packed weights. The experiment measures model quality.

The original checkpoint scored 17.49 perplexity. The compressed runs were paused before any compressed perplexity row was written. The run directories retain the partial measurement files. [Original model row](synthesized/perplexity.csv)

## Protocol

WikiText-2 raw test text is joined with two newlines and tokenized with the checkpoint tokenizer. The evaluator uses nonoverlapping 1,024-token contexts, no extra special tokens, and reports the exponential of mean next-token cross entropy over the entire split. The input contains 304,978 tokens and yields 304,680 scored targets. The same original checkpoint, context, and dataset are used for all methods. [Checkpoint revision](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/d6a5589c239236d22370e2126bbe23d4843c47d9) · [WikiText-2 revision](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1)

The synthesized codec freezes program `3-9796` from the earlier weight-only validation run. It uses eight-weight blocks, a 4-bit exponent, and either a 12- or 16-bit seed. An exact Euclidean index over the complete representable codebook selects the nearest vector and preserves seed-first ties. Reconstruction is made from the packed seed and exponent fields. A scalar decoder checks samples of each payload and a separate exhaustive scorer checks the index on sampled model weights. [Generator search report](../generator-search-smollm2/README.md)

The local SeedLM implementation uses the paper's LFSR with 16-bit seeds, pseudoinverse fit, and 4-bit coefficient and exponent rule. The 3-bpw configuration uses four basis vectors over 12 weights. The 4-bpw configuration uses three basis vectors over eight weights. The authors' published accuracy experiments use other checkpoints. [SeedLM Algorithm 1 and configurations](https://arxiv.org/html/2410.10714v2#S3.SS3)

### Seed search audit

The all-seed encoder checks the continuous projection bound for each of the 65,535 nonzero seeds in Algorithm 1. Every seed that can improve the best reconstruction is fitted, quantized and scored. A 1,024-seed shortlist supplies the initial error bound. The projection comparison includes a conservative float32 tolerance. The bound and an unpruned sweep returned identical packed seeds, coefficients and exponents on all 1,344 audited 3-bit blocks and 2,016 audited 4-bit blocks. The audited spans cover 21 matrices over three layers. A sample of 12 blocks per configuration also agrees with the NumPy reference encoder. [Seed audit](audit/encoder_audit.csv)

The shortlist alone had NMSE 0.11482 at 3 bits and 0.04207 at 4 bits. Unpruned search reached 0.09521 and 0.03708 on the same blocks. The direct codec's index matched exhaustive error and decoded vectors for all 192 audited blocks at each rate. [Direct codebook audit](audit/direct_exactness.csv)

## Reproduce

Use the locked uv environment with the `model` extra. The commands below use the local macOS CPU encoder and the data files described in the [model setup](../../../README.md#computational-codebook-follow-up).

The evaluator defaults to the preserved `winners.json`. Use `--winners` to evaluate a new three-instruction winner and keep its `selection.json` in the same directory.

```sh
uv sync --locked --extra model

clang++ -std=c++20 -O3 -ffp-contract=on -shared -fPIC -pthread \
  experiments/hardware/paper_seedlm_encode.cpp -framework Accelerate -o /tmp/paper-seedlm.dylib

checkpoint=/path/to/smollm2-135m.safetensors
data=/path/to/smollm2-and-wikitext-files
out=runs/generator-model-smollm2
VECLIB_MAXIMUM_THREADS=2 HF_HUB_OFFLINE=1 uv run --locked --extra model python -m experiments.evaluation.generator_model \
  --checkpoint "$checkpoint" --data "$data" --library /tmp/paper-seedlm.dylib \
  --out "$out/synthesized" --methods original synthesized_2 synthesized_2p5 --threads 4

VECLIB_MAXIMUM_THREADS=1 HF_HUB_OFFLINE=1 uv run --locked --extra model python -m experiments.evaluation.generator_model \
  --checkpoint "$checkpoint" --data "$data" --library /tmp/paper-seedlm.dylib \
  --out "$out/seedlm-all-seeds" --methods seedlm_all_seeds_4 seedlm_all_seeds_3 --pool 1024 --threads 4

VECLIB_MAXIMUM_THREADS=1 HF_HUB_OFFLINE=1 uv run --locked --extra model python -m experiments.evaluation.generator_model \
  --checkpoint "$checkpoint" --data "$data" --library /tmp/paper-seedlm.dylib \
  --out "$out/seedlm" --methods seedlm_paper_3 seedlm_paper_4 --pool 1024 --threads 4

VECLIB_MAXIMUM_THREADS=1 uv run --locked --extra model python -m experiments.evaluation.generator_model \
  --checkpoint "$checkpoint" --library /tmp/paper-seedlm.dylib \
  --out "$out/audit" --audit --pool 1024 --threads 4
```
