"""Exact two-tile teacher/student task-loss oracle and local update benchmark."""

import csv
import json
from pathlib import Path
import time

import numpy as np

from experiments.synthesis.trellis import signed_byte


OUT = Path("results/e08-12-low-rate-training/procedural-training/tiny-oracle")


def codebooks():
    seed = np.arange(256, dtype=np.int64)
    discovered = signed_byte(120 * (seed[:, None] >> np.arange(4))).astype(np.float64)
    discovered /= np.sqrt(np.mean(discovered**2))
    digit = (seed[:, None] >> (2*np.arange(4))) & 3
    scalar = np.array([-3., -1., 1., 3.])[digit] / np.sqrt(5.)
    rng = np.random.default_rng(110)
    gaussian = rng.standard_normal((256, 4))
    gaussian /= np.sqrt(np.mean(gaussian**2))
    return {"discovered": discovered, "scalar": scalar, "random_vq": gaussian}


def full_cost(codebook, x, target):
    """Task loss of every two-tile state, including cross-tile interactions."""
    first = x[:, :4] @ codebook.T
    second = x[:, 4:] @ codebook.T
    a = np.mean(first**2, axis=0)
    b = np.mean(second**2, axis=0)
    cross = first.T @ second / len(x)
    y_first = target @ first / len(x)
    y_second = target @ second / len(x)
    return a[:, None] + b[None, :] + 2*cross - 2*y_first[:, None] - 2*y_second[None, :] + np.mean(target**2)


def projected_step(book, current, gradient, eta):
    proposed = current - eta * gradient
    costs = np.sum((book[:, None, :] - proposed[None, :, :])**2, axis=2)
    return costs.argmin(axis=0)


def run(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261127)
    books = codebooks()
    rows = []
    start = time.perf_counter()
    for family, book in books.items():
        for trial in range(12):
            x = rng.normal(size=(96, 8))
            teacher = rng.integers(0, 256, size=2)
            planted = np.r_[book[teacher[0]], book[teacher[1]]]
            if trial % 2:
                planted = planted + 0.35 * rng.normal(size=8)
            target = x @ planted
            costs = full_cost(book, x, target)
            optimum = np.unravel_index(np.argmin(costs), costs.shape)
            current = rng.integers(0, 256, size=2)
            current_loss = float(costs[tuple(current)])
            # Exact full-batch gradient of mean squared task loss.
            residual = x @ np.r_[book[current[0]], book[current[1]]] - target
            gradient = (2 / len(x) * x.T @ residual).reshape(2, 4)
            proposed = projected_step(book, book[current], gradient, 0.1)
            proposed_loss = float(costs[tuple(proposed)])
            # Search a complete one-bit neighbourhood under true task loss.
            neighbors = [(int(current[0]), int(current[1]))]
            neighbors += [(int(current[0] ^ (1 << j)), int(current[1])) for j in range(8)]
            neighbors += [(int(current[0]), int(current[1] ^ (1 << j))) for j in range(8)]
            flip = min(neighbors, key=lambda ij: costs[ij])
            rows.append(dict(family=family, trial=trial, planted=trial%2==0,
                             optimum_loss=float(costs[optimum]), current_loss=current_loss,
                             projected_loss=proposed_loss, flip_loss=float(costs[flip]),
                             projected_optimal=int(tuple(proposed)==optimum),
                             flip_optimal=int(flip==optimum), candidates_exhaustive=65536,
                             candidates_flip=17, candidates_projected=512))
    with (out/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out/"config.json").write_text(json.dumps(dict(seed=20261127, samples=96, trials_per_family=12,
            target_noise=0.35, step_size=0.1, dataset="synthetic Gaussian inputs", runtime_s=time.perf_counter()-start), indent=2)+"\n")
    for family in books:
        group = [r for r in rows if r["family"] == family]
        print(family, "mean regret projected", np.mean([r["projected_loss"]-r["optimum_loss"] for r in group]),
              "flip", np.mean([r["flip_loss"]-r["optimum_loss"] for r in group]))


if __name__ == "__main__":
    run()
