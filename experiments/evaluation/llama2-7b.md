# Llama 2 7B SeedLM reference run

SmolLM2-135M is the quick regression model for codecs and same-checkpoint quality comparisons. Llama 2 7B is the reference model for checking the published SeedLM results. Table 2 reports WikiText-2 perplexity over 166 windows of 2,048 tokens. [SeedLM Table 2](https://arxiv.org/html/2410.10714v2#S4.SS1)

| Llama 2 7B method | Published bits per weight | Published perplexity |
| --- | ---: | ---: |
| Original | 16 | 5.5 |
| SeedLM | 4 | 5.7 |
| SeedLM | 3 | 6.6 |

These are the paper's results. Run the local original model first to check alignment with the published baseline, then run the two all-seed configurations. The local protocol uses a single BOS at the beginning of the joined text, scores the next-token targets within each complete window, and compresses the seven attention and MLP projections in each layer. It groups SeedLM blocks across each contiguous row-major matrix and zero-pads the final block as needed. These are local choices where the paper does not specify the exact implementation. Record any resulting perplexity differences.

## Inputs and build

Run from the repository root on Linux or WSL2. Install a C++20 compiler, `pkg-config`, and the OpenBLAS development package. `--device cuda` requires a driver compatible with the PyTorch build in `uv.lock`. CUDA streams the FP16 weights from host RAM during inference, so the whole model does not need to fit in GPU memory. The checkpoint contains about 13.5 GB of weights. Allow additional system RAM for reconstructed weights, tensor workspaces, and the OS. SeedLM seed search runs on the CPU. [Offload implementation](https://huggingface.co/docs/accelerate/en/package_reference/big_modeling#accelerate.cpu_offload)

Use an authorized local copy of the base Llama 2 7B snapshot with `config.json`, `tokenizer.json`, the SafeTensors index, and both shards. Keep the dataset's raw WikiText-2 Parquet files in a separate directory. [Checkpoint files and access](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main) · [Pinned WikiText-2 files](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1)

```sh
uv sync --locked --extra model
uv run --locked --extra model python -c 'import torch; print(torch.version.cuda, torch.cuda.is_available())'

mkdir -p build
c++ -std=c++20 -O3 -ffp-contract=on -shared -fPIC -pthread -DSEEDLM_USE_CBLAS \
  experiments/hardware/paper_seedlm_encode.cpp $(pkg-config --cflags --libs openblas) \
  -o build/paper-seedlm.so

checkpoint=data/llama-2-7b-hf
data=data/wikitext-2-raw-v1
out=runs/llama2-7b
threads=4
```

## Evaluate

Start with a single 128-token original run to check loading and inference. Use `--device cpu` in the commands if running entirely on the CPU. The perplexity from this smoke test only verifies execution.

```sh
HF_HUB_OFFLINE=1 uv run --locked --extra model python -m experiments.evaluation.generator_model \
  --checkpoint "$checkpoint" --data "$data" --out "$out/smoke" \
  --methods original --device cuda --dtype float16 --threads "$threads" \
  --context 128 --tokens 128 --batch 1 --bos --calibration-tokens 0
```

Use separate directories for the full runs so completed reference rows remain available. The OpenBLAS implementation screens all 65,535 nonzero seeds with the same projection bound used on macOS. `--pool 65535` selects the unpruned CPU search for verification at a higher compute cost. The 1,024-seed shortlist alone is an approximation and has its own method names.

```sh
run_llama() {
  HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    uv run --locked --extra model python -m experiments.evaluation.generator_model \
    --checkpoint "$checkpoint" --data "$data" --device cuda --dtype float16 \
    --threads "$threads" --context 2048 --windows 166 --batch 1 --bos \
    --calibration-tokens 0 "$@"
}

run_llama --out "$out/original" --methods original
run_llama --out "$out/seedlm-4" --methods seedlm_all_seeds_4 \
  --library build/paper-seedlm.so --pool 1024
run_llama --out "$out/seedlm-3" --methods seedlm_all_seeds_3 \
  --library build/paper-seedlm.so --pool 1024
run_llama --out "$out/synthesized" --methods synthesized_2 synthesized_2p5
```

Each complete run writes `perplexity.csv` and `model_config.json`. Keep the original and SeedLM outputs with the reference results for that checkpoint. Compare later experiments to those rows when checkpoint and dataset hashes, projection coverage, dtype, and scoring protocol match. Rate columns measure projection payload, including any padding. Perplexity is evaluated on dense reconstructed weights.
