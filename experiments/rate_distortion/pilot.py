"""Paired controls, spatial granularity sweep, and function-preserving FFN ordering."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.encoding.pursuit import curves
from experiments.generator_algebra.dictionary import dictionary, planted
from experiments.manifests.weights import PROJECTIONS, Weights


FFN = ("gate_proj", "up_proj", "down_proj")
RATES = (0.25, 0.5, 1, 2, 3, 4, 6, 8)


def tile_sample(matrix, shape, maximum, rng):
    height, width = shape
    grid_rows, grid_columns = matrix.shape[0] // height, matrix.shape[1] // width
    positions = rng.choice(grid_rows * grid_columns, min(maximum, grid_rows * grid_columns),
                           replace=False)
    rows, cols = np.divmod(positions, grid_columns)
    return np.stack([
        matrix[row * height:(row + 1) * height, col * width:(col + 1) * width]
        for row, col in zip(rows, cols)
    ])


def control_datasets(matrix, tiles, repeats, injection, rng):
    mean = matrix.mean(dtype=np.float64)
    std = matrix.std(dtype=np.float64)
    yield "original", 0, tiles
    signal = np.stack([planted(tiles.shape[1:], rng) for _ in tiles]).astype(np.float32)
    signal_rms = np.sqrt(np.mean(signal ** 2, axis=(1, 2), keepdims=True))
    centered_rms = np.sqrt(np.mean((tiles - mean) ** 2, axis=(1, 2), keepdims=True))
    yield "planted", 0, (signal * (std or 1) / signal_rms).astype(np.float32)
    yield "injected", 0, (tiles + injection * centered_rms * signal / signal_rms).astype(np.float32)
    for trial in range(repeats):
        sample = rng.choice(matrix.size, tiles.size, replace=False, shuffle=True)
        yield "global_shuffle", trial, matrix.ravel()[sample].reshape(tiles.shape)
        flat = tiles.reshape(len(tiles), -1)
        yield "within_tile_shuffle", trial, rng.permuted(flat, axis=1).reshape(tiles.shape)
        gaussian = rng.standard_normal(tiles.shape, dtype=np.float32)
        yield "gaussian", trial, (gaussian * std + mean).astype(np.float32)


def measurements(tiles, args, metadata, atoms, families):
    energy = np.sum(np.square(tiles, dtype=np.float64))
    for method, setting, payload_bits, errors in curves(
        tiles, atoms, families, args.seeds, max(RATES), args.max_atoms
    ):
        yield {
            **metadata, "method": method, "setting": setting,
            "tiles": len(tiles), "weights": tiles.size,
            "bits_per_weight": payload_bits / tiles[0].size,
            "sse": float(errors.sum()), "energy": float(energy),
            "nmse": float(errors.sum() / energy) if energy else 0.0,
        }


def ordering_experiment(weights, layer, args, rng):
    gate, up, down = [weights.get(layer, name) for name in FFN]
    hidden, width = gate.shape
    if up.shape != gate.shape or down.shape != (width, hidden) or width < 2:
        raise ValueError("Expected three compatible SwiGLU weight matrices")
    # Fit on probes from the first quarter and score only contiguous tiles
    # drawn from the disjoint final three quarters.
    fit_end = width // 4 or 1
    probes = rng.choice(fit_end, min(64, fit_end), replace=False)
    holdout = slice(fit_end, None)
    features = np.column_stack((gate[:, probes], up[:, probes], down[probes, :].T))
    features -= features.mean(axis=0)
    features /= np.maximum(features.std(axis=0), 1e-12)
    u, s, _ = np.linalg.svd(features, full_matrices=False)
    direction = u[:, 0] * s[0]
    direction *= np.sign(direction[np.argmax(np.abs(direction))]) or 1
    fitted = np.argsort(direction, kind="stable")

    x = rng.standard_normal((width, 3))
    def ffn(gate_weights, up_weights, down_weights):
        g, v = gate_weights @ x, up_weights @ x
        silu = g * np.exp(-np.logaddexp(0, -g))
        return down_weights @ (silu * v)

    original = ffn(gate, up, down)
    reordered = ffn(gate[fitted], up[fitted], down[:, fitted])
    relative_error = np.linalg.norm(reordered - original) / max(np.linalg.norm(original), 1e-30)
    if relative_error > 1e-10:
        raise ArithmeticError(f"Shared FFN permutation changed layer {layer} output: {relative_error}")
    print(f"layer {layer}: shared permutation FFN relative error {relative_error:.3g}", flush=True)

    orders = [("original_order", 0, np.arange(hidden)), ("fitted_order", 0, fitted)]
    orders.extend(("random_order", trial, rng.permutation(hidden)) for trial in range(args.repeats))
    for name, full, transpose in zip(FFN, (gate, up, down), (False, False, True)):
        matrix = full[holdout, :].T if transpose else full[:, holdout]
        for size in args.blocks:
            if min(matrix.shape) < size:
                continue
            shape = (size, size)
            atoms, families = dictionary(shape, args.seeds)
            # Sampling positions are paired across every ordering.
            sample_seed = int(rng.integers(2**32))
            for dataset, trial, order in orders:
                ordered = matrix[order]
                if transpose:
                    ordered = ordered.T
                tiles = tile_sample(ordered, shape, args.tiles, np.random.default_rng(sample_seed))
                metadata = {"layer": layer, "projection": name, "granularity": f"{size}x{size}",
                            "dataset": dataset, "trial": trial}
                yield from measurements(tiles, args, metadata, atoms, families)


def run(args):
    rng = np.random.default_rng(args.seed)
    weights = Weights(args.checkpoint) if args.checkpoint else None
    if weights and not weights.layers:
        raise ValueError("No Llama-style transformer layers in the checkpoint")
    layers = args.layers or ([weights.layers[0], weights.layers[len(weights.layers) // 2],
                              weights.layers[-1]] if weights else [0])
    layers = list(dict.fromkeys(layers))
    projections = args.projections or (list(PROJECTIONS) if weights else ["gate_proj"])
    if args.permutation and weights is None:
        raise ValueError("The permutation experiment requires a pretrained checkpoint")

    args.out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(layers=layers, projections=projections, rates=RATES, numpy=np.__version__)
    if weights is not None:
        config["checkpoint_sha256"] = {}
        for path in sorted({entry[0] for entry in weights.tensors.values()}):
            with path.open("rb") as handle:
                config["checkpoint_sha256"][path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    (args.out / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    fields = ("layer", "projection", "granularity", "dataset", "trial", "method", "setting",
              "tiles", "weights", "bits_per_weight", "sse", "energy", "nmse")
    temporary = args.out / "curves.partial.csv.gz"
    with gzip.open(temporary, "wt", newline="", compresslevel=4) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer in layers:
            for projection in projections:
                matrix = (weights.get(layer, projection) if weights else
                          rng.normal(0, 0.03, (256, 256)).astype(np.float32))
                if not np.isfinite(matrix).all():
                    raise ValueError(f"Nonfinite weights in layer {layer} {projection}")
                print(f"layer {layer}: {projection} {matrix.shape}", flush=True)
                geometries = {f"{size}x{size}": (size, size) for size in args.blocks}
                if not args.skip_rows:
                    geometries["row"] = (1, matrix.shape[1])
                if args.whole_matrix:
                    geometries["matrix"] = matrix.shape
                for label, shape in geometries.items():
                    if any(size > limit for size, limit in zip(shape, matrix.shape)):
                        continue
                    atoms, families = dictionary(shape, args.seeds)
                    tiles = tile_sample(matrix, shape, args.tiles, rng)
                    for dataset, trial, values in control_datasets(
                        matrix, tiles, args.repeats, args.injection, rng
                    ):
                        metadata = {"layer": layer, "projection": projection, "granularity": label,
                                    "dataset": dataset, "trial": trial}
                        writer.writerows(measurements(values, args, metadata, atoms, families))
            if args.permutation:
                writer.writerows(ordering_experiment(weights, layer, args, rng))
    temporary.replace(args.out / "curves.csv.gz")

    from experiments.rate_distortion.pilot_report import write_report

    write_report(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="SafeTensors file or local sharded checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--projections", choices=PROJECTIONS, nargs="+")
    parser.add_argument("--blocks", type=int, nargs="*", default=[4, 8, 16, 32, 64],
                        help="Tile side lengths. Pass without sizes to skip the tile sweep.")
    parser.add_argument("--tiles", type=int, default=16, help="Maximum tiles per matrix and granularity")
    parser.add_argument("--repeats", type=int, default=9, help="Independent draws per null control")
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--seeds", type=int, default=16, help="Number of candidate pseudorandom seeds")
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--injection", type=float, default=1.0, help="Planted signal RMS / centred tile RMS")
    parser.add_argument("--skip-rows", action="store_true")
    parser.add_argument("--whole-matrix", action="store_true",
                        help="Also evaluate the full matrix. Memory scales with matrix area times dictionary size.")
    parser.add_argument("--permutation", action="store_true", help="Run shared FFN ordering experiment")
    args = parser.parse_args()
    if not args.blocks and ((args.skip_rows and not args.whole_matrix) or args.permutation):
        parser.error("Choose at least one granularity and supply block sizes for --permutation.")
    if (min(args.blocks + [args.tiles, args.repeats, args.seeds, args.max_atoms]) < 1
            or args.seeds > 200 or not np.isfinite(args.injection) or args.injection <= 0):
        parser.error("Use positive settings, at most 200 candidate seeds and a finite injection strength.")
    if len(set(args.blocks)) != len(args.blocks) or (
        args.projections and len(set(args.projections)) != len(args.projections)
    ):
        parser.error("Block sizes and projections must be unique.")
    run(args)


if __name__ == "__main__":
    main()
