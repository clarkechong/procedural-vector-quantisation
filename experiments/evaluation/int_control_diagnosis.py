"""Diagnose the historical whole-matrix odd-level INT8 control.

This does not alter the earlier model-level run.  It compares identical source
weights with independent FP32 dequantization and several scale policies.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.manifests.weights import Weights


def quantize(values, levels, iterations=3, initial=None):
    x = np.asarray(values, dtype=np.float32)
    levels = np.asarray(levels, dtype=np.float32)
    scale = (float(np.sqrt(np.mean(x.astype(np.float64) ** 2) /
                           np.mean(levels.astype(np.float64) ** 2)))
             if initial is None else float(initial))
    for _ in range(iterations):
        codes = np.abs(x[..., None] / scale - levels).argmin(-1) if len(levels) <= 16 else None
        if codes is None:
            # Nearest level for the regular odd or zero-inclusive unit grid.
            step = float(levels[1] - levels[0])
            codes = np.clip(np.rint((x / scale - levels[0]) / step),
                            0, len(levels)-1).astype(np.uint8)
        chosen = levels[codes]
        scale = max(float(np.sum(x * chosen, dtype=np.float64) /
                          np.sum(chosen.astype(np.float64) ** 2)), 1e-12)
    step = float(levels[1] - levels[0])
    codes = np.clip(np.rint((x / scale - levels[0]) / step),
                    0, len(levels)-1).astype(np.uint8)
    restored = levels[codes] * np.float32(scale)
    return restored, codes, scale


def quality(x, y):
    diff = x.astype(np.float64) - y.astype(np.float64)
    return float(np.sum(diff ** 2) / np.sum(x.astype(np.float64) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("data/smollm2-135m.safetensors"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/e13-18-activation-landscape/int8-diagnosis"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = Weights(args.checkpoint)
    odd = np.arange(-255, 256, 2, dtype=np.float32)
    zero = np.arange(-128, 128, dtype=np.float32)
    rows = []
    for layer, name in ((0, "q_proj"), (0, "o_proj"), (17, "q_proj"),
                        (17, "k_proj"), (12, "down_proj")):
        x = weights.get(layer, name).astype(np.float32)
        for label, levels, steps in (("odd_3", odd, 3), ("odd_50", odd, 50),
                                     ("zero_3", zero, 3), ("zero_50", zero, 50)):
            y, codes, scale = quantize(x, levels, steps)
            # Independent scalar lookup dequantizer; catches sign/bit order.
            decoded = np.take(levels, codes) * np.float32(scale)
            assert np.array_equal(y, decoded)
            rows.append(dict(layer=layer, projection=name, method=label,
                             nmse=quality(x, y), scale=scale,
                             zero_fraction=float(np.mean(y == 0)),
                             clipped_fraction=float(np.mean((codes == 0) | (codes == len(levels)-1)))))
        # A row-wise zero-inclusive positive control holds bitwidth and levels fixed.
        restored = np.empty_like(x)
        for i in range(x.shape[0]):
            restored[i], _, _ = quantize(x[i], zero, 3)
        rows.append(dict(layer=layer, projection=name, method="zero_row_3",
                         nmse=quality(x, restored), scale="per_row",
                         zero_fraction=float(np.mean(restored == 0)),
                         clipped_fraction=""))
        print(layer, name, rows[-5:], flush=True)
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "config.json").write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint), layers=[0, 12, 17],
        projection_types=["q_proj", "k_proj", "o_proj", "down_proj"],
        precision="source BF16 converted to FP32; encoding and dequantization FP32",
        notes="Same 8-bit payload throughout; row scales cost 32 bits per output row."
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
