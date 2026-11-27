"""Evaluate pinned SmoothQuant fake W8A8 on local Llama-2 and WikiText-2 files."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from experiments.evaluation.seedlm_model import (
    digest, inference_weights, load_model, perplexity, token_ids,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/llama-2-7b-hf"))
    parser.add_argument("--data", type=Path, default=Path("data/wikitext-2-raw-v1"))
    parser.add_argument("--scales", type=Path, default=Path("data/smoothquant-scales/llama-2-7b.pt"))
    parser.add_argument("--upstream", type=Path, default=Path("/tmp/smoothquant-official"))
    parser.add_argument("--out", type=Path, default=Path("runs/llama2-7b/smoothquant"))
    parser.add_argument("--method", choices=("original", "w8a8", "smooth_w8a8"), required=True)
    parser.add_argument("--windows", type=int, default=166)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    from smoothquant.fake_quant import quantize_model
    from smoothquant.smooth import smooth_lm

    torch.set_num_threads(4)
    model = load_model(args.checkpoint, args.checkpoint, getattr(torch, args.dtype))
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    # Upstream AutoTokenizer adds the Llama BOS token by default.
    tokens = token_ids(args.data, "test", tokenizer, model.config.bos_token_id)[: args.windows * args.context]
    assert len(tokens) == args.windows * args.context
    started = time.perf_counter()
    device = torch.device("cuda")
    model.to(device)
    if args.method == "smooth_w8a8":
        scales = torch.load(args.scales, map_location="cpu", weights_only=True)
        smooth_lm(model, scales, alpha=0.85)
    if args.method != "original":
        model = quantize_model(model, weight_quant="per_channel", act_quant="per_token",
                               quantize_bmm_input=True)
    setup_seconds = time.perf_counter() - started
    with torch.inference_mode(), inference_weights(model, device):
        nll, scored = perplexity(model, tokens, args.context, 1, device)
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(method=args.method, ppl=float(np.exp(nll)), nll=nll, scored_tokens=scored,
                  windows=args.windows, context=args.context, dtype=args.dtype,
                  setup_seconds=setup_seconds, upstream_commit="c61476d728e42ae0d8a35e7e78494edcac3237b5",
                  checkpoint_index_sha256=digest(args.checkpoint / "model.safetensors.index.json"),
                  test_sha256=digest(args.data / "test-00000-of-00001.parquet"),
                  scale_sha256=digest(args.scales) if args.method == "smooth_w8a8" else None,
                  note="Pinned upstream SmoothQuant transforms and fake quantizer; local equivalent evaluator with one BOS")
    (args.out / f"{args.method}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
