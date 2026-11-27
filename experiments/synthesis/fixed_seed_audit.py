"""Enumerate the eight-bit multiplier-120 seed map and its distinct extensions."""

import csv
from pathlib import Path

import numpy as np

from experiments.synthesis.trellis import signed_byte


def audit(out=Path("results/e13-18-activation-landscape/fixed_seed_audit.csv")):
    out.parent.mkdir(parents=True, exist_ok=True)
    seed = np.arange(256, dtype=np.int64)[:, None]
    rows = []
    for width in (4, 8, 16, 32):
        i = np.arange(width)[None, :]
        variants = {
            "literal_shift": signed_byte(120 * (seed >> i)),
            "cyclic_shift": signed_byte(120 * (seed >> (i % 8))),
            "window_L5_K2_zero_padded": signed_byte(120 * ((seed >> (2*i)) & 31)),
        }
        for name, book in variants.items():
            book = np.asarray(book, dtype=np.float64)
            # A coordinate is an identical function only if all 256 values match.
            coordinate_functions = {book[:, j].tobytes() for j in range(width)}
            influence = np.asarray([
                np.any(book != book[(np.arange(256) ^ (1 << bit)), :], axis=0)
                for bit in range(8)])
            rows.append(dict(width=width, variant=name,
                             structurally_zero_fraction=float(np.mean(np.all(book == 0, axis=0))),
                             distinct_coordinate_functions=len(coordinate_functions),
                             unique_codewords=len(np.unique(book, axis=0)),
                             matrix_rank=int(np.linalg.matrix_rank(book)),
                             active_seed_bit_coordinate_pairs=int(influence.sum()),
                             coordinate_rms=";".join(f"{v:.6g}" for v in
                                                     np.sqrt(np.mean(book**2, axis=0)))))
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == "__main__":
    for row in audit():
        print({k: v for k, v in row.items() if k != "coordinate_rms"})
