"""Experiment 04B: one-projection functional scaling/weighted-trellis pilot."""

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.evaluation.seedlm_model import load_model, perplexity, token_ids
from experiments.synthesis.trellis import linear_map
from experiments.synthesis.weighted_trellis import encode, refit_scale
from tokenizers import Tokenizer


OUT = Path("results/e04-07-reference-optimization/activation-ptq/pilot")


def quantize(matrix, metric, mapping, scale):
    width = matrix.shape[1]
    if metric.shape != (width,):
        raise ValueError(f"matrix {matrix.shape}, metric {metric.shape}")
    reconstructed = np.empty_like(matrix)
    for row in range(len(matrix)):
        reconstructed[row] = encode(matrix[row], metric, 5, 2, mapping*scale)[0]
    return reconstructed


def fit_scale(matrix, metric, mapping):
    # Fixed calibration spans; exact assignments conditional on each scale.
    chosen = np.linspace(0, len(matrix)-1, 8, dtype=int)
    starts = np.linspace(0, matrix.shape[1]-64, 8, dtype=int)
    spans = np.stack([matrix[r, s:s+64] for r, s in zip(chosen, starts)])
    weights = np.stack([metric[s:s+64] for s in starts])
    base = np.sqrt(np.mean(spans**2) / np.mean(mapping**2))
    best = (np.inf, base)
    for factor in (.7, 1., 1.4):
        scale = base * factor
        for _ in range(2):
            numerator = denominator = 0.
            for row, h, target in zip(spans, weights, spans):
                decoded = encode(target, h, 5, 2, mapping*scale)[0] / scale
                numerator += np.sum(h*target*decoded)
                denominator += np.sum(h*decoded**2)
            if denominator:
                scale = max(numerator/denominator, 1e-12)
        error = sum(encode(row, h, 5, 2, mapping*scale)[2] for row, h in zip(spans, weights))
        if error < best[0]:
            best = (error, scale)
    return float(best[1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file("data/smollm2/tokenizer.json")
    all_tokens = token_ids(Path("data/smollm2"), "validation", tokenizer)
    calibration = all_tokens[:512]
    validation = all_tokens[512:1024]
    # Write before any model run. Prior experiments may already have used validation.
    (OUT/"manifest.json").write_text(json.dumps(dict(dataset="local WikiText-2 validation parquet",
        calibration_token_offsets=[0,511], validation_token_offsets=[512,1023],
        layer=15, projection="down_proj", context=128,
        note="Adaptation pilot; validation is not a blind final-test set"), indent=2)+"\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path("data/smollm2-135m.safetensors"), Path("data/smollm2"), torch.bfloat16).to(device)
    module = model.model.layers[15].mlp.down_proj
    captured = {"calibration": [], "validation": []}
    phase = ["calibration"]
    def hook(_, args):
        captured[phase[0]].append(args[0].detach().float().cpu().reshape(-1, args[0].shape[-1]).numpy())
    handle = module.register_forward_pre_hook(hook)
    with torch.no_grad():
        for tokens, name in ((calibration, "calibration"), (validation, "validation")):
            phase[0] = name
            for block in tokens.split(128):
                model(block[None].to(device), use_cache=False)
    handle.remove()
    xcal = np.concatenate(captured["calibration"])
    xval = np.concatenate(captured["validation"])
    h = np.mean(xcal.astype(np.float64)**2, axis=0)
    weight = module.weight.detach().float().cpu().numpy().astype(np.float64)
    original = module.weight.detach().clone()
    dense_nll, count = perplexity(model, validation, 128, 1, str(device))
    teacher = xval @ weight.T
    mapping = linear_map(5, 120)
    configs = [("A", 0.), ("C", 0.)] + [(family, alpha) for family in ("B", "D")
                                             for alpha in (-.5, 0., .25, .5, .75, 1.)]
    rows = []
    start = time.perf_counter()
    for family, alpha in configs:
        sqrt_h = np.sqrt(h)
        scale_vector = np.power(sqrt_h+1e-8, alpha)
        scale_vector /= np.exp(np.mean(np.log(scale_vector)))
        scale_vector = np.clip(scale_vector, 1/16, 16)
        transformed = family in ("B", "D")
        target = weight*scale_vector[None] if transformed else weight
        metric = h/scale_vector**2 if family == "D" else h if family == "C" else np.ones_like(h)
        matrix_scale = fit_scale(target, metric, mapping)
        reconstructed = quantize(target, metric, mapping, matrix_scale)
        restored = reconstructed/scale_vector[None] if transformed else reconstructed
        activation = xval @ restored.T
        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(restored).to(device=device, dtype=original.dtype))
            nll, tokens = perplexity(model, validation, 128, 1, str(device))
            module.weight.copy_(original)
        delta = restored-weight
        rows.append(dict(family=family, alpha=alpha, matrix_scale=matrix_scale,
            weight_nmse=float(np.sum(delta**2)/np.sum(weight**2)),
            activation_nmse=float(np.sum((teacher-activation)**2)/np.sum(teacher**2)),
            diag_proxy=float(np.sum(delta**2*h[None])/np.sum(weight**2*h[None])),
            nll=nll, excess_nll=nll-dense_nll, ppl=float(np.exp(nll)),
            valid_targets=tokens, payload_bpw=(5+2*(weight.shape[1]-1))/weight.shape[1],
            stored_bpw=(8*((5+2*(weight.shape[1]-1)+7)//8)*weight.shape[0]
                        +32+(32*len(h) if transformed else 0))/weight.size,
            scale_bytes=4, transform_bytes=4*len(h) if transformed else 0))
    with (OUT/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUT/"config.json").write_text(json.dumps(dict(seed=20261127, model="SmolLM2-135M",
          dtype="bfloat16", dense_nll=dense_nll, dense_ppl=float(np.exp(dense_nll)),
          valid_targets=count, input_channel_count=len(h), calibration_tokens=len(calibration),
          validation_tokens=len(validation), epsilon=1e-8, clip=[1/16,16],
          runtime_s=time.perf_counter()-start, device=str(device)), indent=2)+"\n")
    print(json.dumps(dict(dense_nll=dense_nll, best=min(rows, key=lambda r:r["nll"])), indent=2))


if __name__ == "__main__":
    main()
