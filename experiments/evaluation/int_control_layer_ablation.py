"""Single-layer INT8 intervention on SmolLM2 with the original test protocol."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from experiments.evaluation.int_controls import UniformScalar
from experiments.evaluation.seedlm_model import load_model, perplexity, submodules, token_ids
from experiments.manifests.weights import Weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 11, 28])
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = Path("data/smollm2-135m.safetensors")
    assets = Path("data/smollm2")
    out = Path("results/e13-18-activation-landscape/int8-layer-ablation")
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint, assets, torch.float32).to("cuda")
    tokens = token_ids(assets, "test", Tokenizer.from_file(str(assets / "tokenizer.json")))[:256]
    source = Weights(checkpoint)
    codec = UniformScalar(8, 50)
    rows = []
    with torch.inference_mode():
        baseline, count = perplexity(model, tokens, 128, 1, "cuda")
        for layer in args.layers:
            changes = []
            for _, name, module in submodules(model, [layer]):
                old = module.weight
                x = source.get(layer, name)
                y, _ = codec.encode_matrix(x)
                quantized = torch.nn.Parameter(torch.from_numpy(y).to("cuda"), requires_grad=False)
                module.weight = quantized
                changes.append((module, old, quantized, name, x, y))
            nll, _ = perplexity(model, tokens, 128, 1, "cuda")
            rows.append(dict(layer=layer, projection="all", nll=nll,
                             delta_nll=nll-baseline, nmse=""))
            for module, old, quantized, name, x, y in changes:
                for other, earlier, _, *_ in changes:
                    if other is not module:
                        other.weight = earlier
                single, _ = perplexity(model, tokens, 128, 1, "cuda")
                rows.append(dict(layer=layer, projection=name, nll=single,
                                 delta_nll=single-baseline,
                                 nmse=float(np.sum((x-y).astype(np.float64)**2) /
                                            np.sum(x.astype(np.float64)**2))))
                for other, _, q_other, *_ in changes:
                    if other is not module:
                        other.weight = q_other
            for module, old, *_ in changes:
                module.weight = old
            print(layer, nll-baseline, flush=True)
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / "config.json").write_text(json.dumps(dict(
        baseline_nll=baseline, scored_targets=count, context=128, windows=2,
        layers=args.layers,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        quantizer="one FP32 scale per matrix, odd signed 8-bit levels, 50 coordinate-descent updates",
        torch=torch.__version__, hip=torch.version.hip), indent=2) + "\n")


if __name__ == "__main__":
    main()
