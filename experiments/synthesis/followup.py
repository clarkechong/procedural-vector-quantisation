"""Experiment 02: scale fitting, geometry, and a scalar-representable grammar.

Run with ``python -m experiments.synthesis.followup --device cuda``.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.generator_algebra import bitvector
from experiments.manifests.weights import PROJECTIONS, Weights
from experiments.synthesis import miniature


OUT = Path("results/e02-codebook-followup")
EXTRA_CONSTANTS = (4, 7, 8, 16, 31, 0x7f, 0x8000, 0x7f80, 0x007f, 0x3c00, 0x3d00)


def final_sample(checkpoint, old_manifest):
    """Draw only from rows absent in both earlier splits of every matrix."""
    rng = np.random.default_rng(miniature.SEED + 30)
    used = {}
    for entry in old_manifest:
        used.setdefault((entry["layer"], entry["projection"]), set()).add(entry["row"])
    weights = Weights(checkpoint)
    matrices = [(layer, proj) for layer in miniature.LAYERS for proj in PROJECTIONS]
    tiles, manifest = [], []
    for index, key in enumerate(matrices):
        matrix = weights.get(*key)
        remaining = np.setdiff1d(np.arange(matrix.shape[0]),
                                 np.fromiter(used.get(key, ()), dtype=np.int64))
        if not len(remaining):
            raise ValueError(f"No fresh rows for {key}")
        count = miniature.TILES // len(matrices) + (index < miniature.TILES % len(matrices))
        rows = rng.choice(remaining, count, replace=True)
        starts = rng.integers(0, matrix.shape[1] // 4, count) * 4
        tiles.append(matrix[rows[:, None], starts[:, None] + np.arange(4)])
        manifest.extend(dict(layer=key[0], projection=key[1], row=int(row), start=int(start))
                        for row, start in zip(rows, starts))
    return np.concatenate(tiles).astype(np.float32), manifest


def optimize_scale(books, targets, device, initial, iterations=5):
    """Alternating exact seed assignment and nonnegative least-squares scale.

    Three starts reduce local-optimum sensitivity. The fitted scale is shared
    across all tiles for each codebook. This is not a global joint certificate.
    """
    raw = torch.as_tensor(np.asarray(books), dtype=torch.float32, device=device)
    points = torch.as_tensor(targets, dtype=torch.float32, device=device)
    energy = points.double().square().sum().item()
    norm = raw.square().sum(2)
    cross = torch.bmm(raw, points.T.expand(len(raw), -1, -1))
    best_error = torch.full((len(raw),), float("inf"), device=device, dtype=torch.float64)
    best_scale = torch.zeros(len(raw), device=device, dtype=torch.float64)
    for multiple in (0.25, 1.0, 4.0):
        scale = torch.full((len(raw),), initial * multiple, device=device)
        for _ in range(iterations):
            choice = (scale[:, None, None] ** 2 * norm[:, :, None]
                      - 2 * scale[:, None, None] * cross).argmin(1)
            xy = cross.gather(1, choice[:, None, :]).double().sum((1, 2))
            xx = norm.gather(1, choice).double().sum(1)
            updated = (xy / xx.clamp_min(1e-30)).clamp_min(0).float()
            scale = torch.where(xx > 0, updated, scale)
        choice = (scale[:, None, None] ** 2 * norm[:, :, None]
                  - 2 * scale[:, None, None] * cross).argmin(1)
        selected = raw.gather(1, choice[:, :, None].expand(-1, -1, 4))
        residual = points[None] - scale[:, None, None] * selected
        error = residual.double().square().sum((1, 2))
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale.double(), best_scale)
    return (best_error / energy).cpu().numpy(), best_scale.cpu().numpy()


def geometry(book):
    values = book.astype(np.float64)
    centered = values - values.mean(0)
    cov = centered.T @ centered / len(values)
    rms = np.sqrt(np.mean(values ** 2, axis=0))
    eigen = np.linalg.eigvalsh(cov)
    edges = np.stack([np.linalg.norm(values[np.arange(256) ^ (1 << bit)] - values, axis=1)
                      for bit in range(8)])
    return dict(coord_rms_cv=float(rms.std() / rms.mean()) if rms.mean() else None,
                coord_zero_fraction=float(np.mean(values == 0)),
                covariance_condition=float(eigen[-1] / max(eigen[0], 1e-12)),
                covariance_eigenvalues=eigen.tolist(),
                avalanche_mean_l2=float(edges.mean()),
                avalanche_cv=float(edges.std() / edges.mean()) if edges.mean() else None)


def scalar_program(seed, levels):
    """Generic SHL, SHR, AND, LUT4 primitives; no packed-field special case."""
    index = np.arange(4, dtype=np.uint32)[None, :]
    seed = np.asarray(seed, dtype=np.uint32)[:, None]
    r0 = bitvector.vector_instruction(bitvector.Instruction("SHL", ("i", 1)), {"i": index})
    r1 = bitvector.vector_instruction(bitvector.Instruction("SHR", ("s", "r0")),
                                       {"s": seed, "r0": r0})
    r2 = bitvector.vector_instruction(bitvector.Instruction("AND", ("r1", 3)), {"r1": r1})
    return np.asarray(levels, dtype=np.float32)[r2]


def target_geometry(cal, hold):
    result = {}
    for name, array in (("calibration", cal), ("heldout", hold)):
        centered = array.astype(np.float64) - array.mean(0, dtype=np.float64)
        cov = centered.T @ centered / len(centered)
        eigen, vectors = np.linalg.eigh(cov)
        result[name] = {"mean": array.mean(0).tolist(), "covariance": cov.tolist(),
                        "pca_eigenvalues": eigen[::-1].tolist(),
                        "pca_top_vector": vectors[:, -1].tolist(),
                        "isotropy_ratio": float(eigen[0] / eigen[-1])}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/smollm2-135m.safetensors")
    parser.add_argument("--previous", default="results/e01-miniature-codebook")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    start = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    previous = Path(args.previous)
    old = json.loads((previous / "summary.json").read_text())
    old_manifest = json.loads((previous / "sample_manifest.json").read_text())
    samples, fresh_manifest = miniature.sample(args.checkpoint)
    cal, hold = samples["calibration"], samples["heldout"]
    final, manifest = final_sample(args.checkpoint, old_manifest)
    assert not set((e["layer"], e["projection"], e["row"]) for e in manifest) & set(
        (e["layer"], e["projection"], e["row"]) for e in old_manifest)
    assert fresh_manifest == old_manifest
    (out / "final_manifest.json").write_text(json.dumps(manifest) + "\n")

    ranked = list(csv.DictReader((previous / "ranked.csv").open()))
    representatives = {int(row["id"]): row for row in ranked}
    seen = set()
    batch, rows = [], []
    fixed = old["global_scale"]

    def flush():
        if not batch:
            return
        scores, scales = optimize_scale(batch, cal, args.device, fixed)
        for record, score, scale in zip(rows[-len(batch):], scores, scales):
            record["fitted_calibration_nmse"] = float(score)
            record["fitted_scale"] = float(scale)
        batch.clear()

    for program in miniature.candidate_programs():
        book = miniature.code(program)
        key = book.tobytes()
        if key in seen:
            continue
        seen.add(key)
        # The previous search assigns IDs in first-seen order.
        prior = representatives[len(rows)]
        assert prior["program"] == bitvector.expression(program)
        record = {"id": len(rows), "program": prior["program"],
                  "instructions": int(prior["instructions"]),
                  "fixed_calibration_nmse": float(prior["original_nmse"]), **geometry(book)}
        rows.append(record)
        batch.append(book)
        if len(batch) == 16:
            flush()
    flush()
    rows.sort(key=lambda row: row["fitted_calibration_nmse"])
    # Only calibration chooses programs and scales. The original held-out set
    # remains a comparison set; the new disjoint-row final set is evaluated once.
    winners = {}
    for budget in (1, 2):
        winner = next(row for row in rows if row["instructions"] <= budget)
        program = next(p for p in miniature.candidate_programs()
                       if bitvector.expression(p) == winner["program"])
        book = miniature.code(program)
        winner["heldout_nmse"] = miniature.score_one(book, hold, winner["fitted_scale"], args.device)
        winner["final_nmse"] = miniature.score_one(book, final, winner["fitted_scale"], args.device)
        winners[str(budget)] = winner
    # A scalar reconstruction is a 4-instruction program plus four FP16 levels.
    levels = np.asarray(old["baselines"]["lloyd_max_scalar"]["levels"], dtype=np.float32)
    scalar_book = scalar_program(np.arange(256), levels)
    scalar = {"program": "r0=SHL(i,1); r1=SHR(s,r0); r2=AND(r1,3); r3=LUT4(r2)",
              "calibration_nmse": miniature.score_one(scalar_book, cal, 1, args.device),
              "heldout_nmse": miniature.score_one(scalar_book, hold, 1, args.device),
              "final_nmse": miniature.score_one(scalar_book, final, 1, args.device),
              "shared_bytes": 4 * 3 + 8, "description": "four 3-byte instructions and four FP16 table entries"}
    with (out / "optimized_ranked.csv").open("w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {"scale_search": "three initial scales, five alternating seed/scale updates; local optimization",
              "distinct_codebooks": len(rows), "winners": winners,
              "scalar_program": scalar, "target_geometry": target_geometry(cal, hold),
              "elapsed_seconds": time.monotonic() - start,
              "fresh_final_tiles": len(final), "additional_constants_proposed": EXTRA_CONSTANTS}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"winners": winners, "scalar_program": scalar,
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
