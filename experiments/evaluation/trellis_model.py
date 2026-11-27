"""Model-level smoke test of the frozen five-bit overlapping-window codec."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.evaluation.seedlm_model import add_evaluation_arguments, run
from experiments.synthesis.trellis import encode, linear_map


class WindowEncoder:
    def __init__(self, stride=1, calibration_rows=8, calibration_length=64):
        if stride not in (1, 2):
            raise ValueError("Only one- or two-bit strides are supported")
        self.stride = stride
        self.calibration_rows = calibration_rows
        self.calibration_length = calibration_length
        self.mapping = linear_map(5, 120)
        self.matrix_count = 0

    def encode_matrix(self, matrix):
        from experiments.synthesis.trellis_experiment import fit_scale

        rows, width = matrix.shape
        if width < self.calibration_length:
            raise ValueError("Matrix rows are shorter than the calibration span")
        # Deterministic sample from this matrix; scale is shared by every row.
        chosen = np.linspace(0, rows - 1, min(rows, self.calibration_rows), dtype=int)
        starts = np.linspace(0, width - self.calibration_length, len(chosen), dtype=int)
        spans = np.asarray([matrix[row, start:start + self.calibration_length]
                            for row, start in zip(chosen, starts)], dtype=np.float64)
        scale = float(np.float32(fit_scale(spans, 5, self.stride, self.mapping,
                                          starts=(1.0,), passes=2)))
        lookup = self.mapping * scale
        restored = np.empty_like(matrix, dtype=np.float32)
        bytes_per_row = (5 + self.stride * (width - 1) + 7) // 8
        payload = bytearray(np.float32(scale).tobytes())
        for row in range(rows):
            reconstruction, bits, _ = encode(matrix[row], 5, self.stride, lookup)
            restored[row] = reconstruction
            payload.extend(bits.to_bytes(bytes_per_row, "little"))
        self.matrix_count += 1
        return restored, bytes(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/smollm2-135m.safetensors"))
    parser.add_argument("--data", type=Path, default=Path("data/smollm2"))
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/results/trellis-followup/model-smoke"))
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--calibration-tokens", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=("original", "window_120"),
                        default=("original", "window_120"))
    parser.add_argument("--stride", type=int, choices=(1, 2), default=1)
    add_evaluation_arguments(parser)
    args = parser.parse_args()
    args.seeds = 255
    args.shortlist = 8
    args.neighborhood = 192
    if args.tokens < 2 or args.context < 2 or args.threads < 1:
        parser.error("Invalid token, context, or thread count")
    metadata = dict(codec=f"five-bit windows, {args.stride} new bits per weight, byte-aligned rows",
                    map="int8(120*state)", boundary="open", scale="per matrix, 8 training spans",
                    note="Smoke test only; shared scale included in each matrix payload")
    run(args, {"window_120": lambda: WindowEncoder(stride=args.stride)}, metadata)
    (args.out / "codec.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
