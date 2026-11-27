"""Audit the exact procedural books used by the TinyStories and MNIST QAT code."""

import csv
from pathlib import Path

import numpy as np

from experiments.training.low_rate_qat import make_book


def audit(out=Path("results/e19-21-qtip-training/training_map_audit.csv")):
    rows = []
    for width in (4, 8, 16, 32):
        for family in ("mul120", "qtip_1mad", "hash32", "random_vq"):
            book = make_book(family, width).astype(np.float64)
            zero = np.all(np.abs(book) < 1e-10, axis=0)
            coordinate_functions = len({book[:, j].tobytes() for j in range(width)})
            unique_words = len(np.unique(book, axis=0))
            # Count exact coordinate duplicates and constant coordinates after normalization.
            constant = np.max(book, axis=0) - np.min(book, axis=0) < 1e-10
            rms = np.sqrt(np.mean(book ** 2, axis=0))
            rows.append(dict(family=family, width=width, payload_bpw=8 / width,
                             structurally_zero_coordinates=int(zero.sum()),
                             constant_coordinates=int(constant.sum()),
                             distinct_coordinate_functions=coordinate_functions,
                             unique_codewords=unique_words,
                             rank=int(np.linalg.matrix_rank(book)),
                             min_coordinate_rms=float(rms.min()), max_coordinate_rms=float(rms.max()),
                             map_source="experiments.training.low_rate_qat.make_book"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    audit()
