"""Measure exact SeedLM fitting, alternative vector codebooks, and trellis baselines."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.encoding.codebooks import (
    additive, computed_vectors, direct, hadamard_rows, lattice_quantize, lattice_vectors,
    permutation_vectors, signed_permutation, trellis_row,
)
from experiments.encoding.discrete_oracle import exact
from experiments.encoding.seedlm import PUBLISHED, basis, check_reconstruction, search, seed_ids
from experiments.manifests.sampling import samples
from experiments.manifests.weights import Weights


def projection_metrics(out, split, block_count):
    options = ("lfsr", "xorshift", "splitmix", "binary", "ternary", "hadamard")
    with (out / "subspaces.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("family", "layer", "projection", "rule", "candidates", "blocks",
                         "nmse", "mean_family_rank", "unique_bases"))
        for family in options:
            print(f"Subspace search: {family}", flush=True)
            config = replace(PUBLISHED[1], family=family)
            seeds = seed_ids(config)
            bases = basis(config, seeds)
            rank = np.linalg.matrix_rank(bases.astype(np.float64)).mean()
            unique = len(np.unique(bases.reshape(len(seeds), -1), axis=0))
            matrices = [values[0].reshape(-1, config.block)[:block_count]
                        for values in split.values()]
            blocks = np.concatenate(matrices)
            energy = np.square(blocks, dtype=np.float64).reshape(len(split), block_count, 8).sum(axis=(1, 2))
            for count, rule, result in search(config, blocks, ("paper", "refine", "float"),
                                               prefixes=(1, 15, 255, 4095)):
                errors = result[0].reshape(len(split), block_count).sum(axis=1, dtype=np.float64)
                for (layer, name), error, denominator in zip(split, errors, energy):
                    writer.writerow((family, layer, name, rule, count, block_count,
                                     error / denominator, rank, unique))
                if rule != "float" and count == len(seeds):
                    check_reconstruction(config, blocks, result)
            handle.flush()


def oracle_metrics(out, split, block_count):
    blocks = np.concatenate([values[0].reshape(-1, 8)[:block_count] for values in split.values()])
    with (out / "oracle.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("family", "layer", "projection", "block_index", "packed_word_le", "energy", "paper_sse",
                         "refine_sse", "bounded_sse", "uncapped_sse", "free_scale_sse",
                         "float_sse", "visited_seeds"))
        for family in ("lfsr", "binary", "ternary"):
            print(f"Exact oracle: {family}", flush=True)
            config = replace(PUBLISHED[1], family=family)
            fitted = {rule: result for _, rule, result in search(config, blocks, ("paper", "refine"))}
            optimum = exact(config, blocks, fitted["refine"])
            packed_fields = tuple(np.asarray(value) for value in zip(*(row[:4] for row in optimum)))
            payload = check_reconstruction(config, blocks, packed_fields)
            (out / f"oracle_{family}.bin").write_bytes(payload)
            for i, (bounded, _, _, _, floating, uncapped, free_scale, visited) in enumerate(optimum):
                layer, name = list(split)[i // block_count]
                writer.writerow((family, layer, name, i % block_count, payload[4 * i:4 * i + 4].hex(),
                                 blocks[i].astype(np.float64) @ blocks[i],
                                 fitted["paper"][0][i], fitted["refine"][0][i], bounded,
                                 uncapped, free_scale, floating, visited))
            handle.flush()


def other_metrics(out, screen, split):
    direct6 = computed_vectors(np.arange(1 << 20), 6)
    direct8 = computed_vectors(np.arange(1 << 20), 8)
    first = computed_vectors(np.arange(1 << 14), 8, 0xA419F3)
    second = computed_vectors(np.arange(1 << 14), 8, 0xCD87F2)
    first[0], second[0] = 0, 0
    permuted = permutation_vectors()
    lattice = lattice_vectors()
    assert lattice.shape == (65536, 8)
    with (out / "other_codebooks.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "layer", "projection", "weights", "payload_bpw",
                         "scale_metadata_bits_per_matrix", "nmse", "encode_seconds"))
        for (layer, name), (values, *_) in split.items():
            print(f"Vector and trellis baselines: {layer}/{name}", flush=True)
            # Every codec gets identical 192-weight held-out spans.
            blocks = values[0].reshape(-1, 8)
            scale = np.float32(np.float16(np.sqrt(np.mean(np.square(screen[layer, name][0], dtype=np.float64)))))
            operations = (
                ("direct_c6", 4, 0, lambda: direct(direct6, values[0].reshape(-1, 6))),
                ("direct_c8", 3, 0, lambda: direct(direct8, blocks)),
                ("additive", 4, 0, lambda: additive(first, second, blocks)),
                ("signed_permutation", 4, 0, lambda: signed_permutation(permuted, blocks)),
                ("e8_shaped", 2, 16, lambda: lattice_quantize(blocks, lattice, scale)),
                ("e8_residual", 4, 16, lambda: e8_residual(blocks, lattice, scale)),
                ("trellis_1mad", 3, 16, lambda: [trellis_row(row / scale, 3) * scale for row in values]),
                ("trellis_1mad", 4, 16, lambda: [trellis_row(row / scale, 4) * scale for row in values]),
            )
            target = values[0].ravel()
            energy = np.square(target, dtype=np.float64).sum()
            for method, rate, overhead, operation in operations:
                started = time.perf_counter()
                restored = np.asarray(operation()).ravel()[:len(target)]
                elapsed = time.perf_counter() - started
                error = np.square(restored - target, dtype=np.float64).sum() / energy
                writer.writerow((method, layer, name, len(target), rate, overhead, error, elapsed))
            handle.flush()


def e8_residual(blocks, lattice, scale):
    initial = lattice_quantize(blocks, lattice, scale)
    return initial + lattice_quantize(blocks - initial, lattice, scale / np.float32(4))


def rotated_metrics(out, screen, split):
    lattice = lattice_vectors()
    with (out / "rotated_baselines.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "layer", "projection", "weights", "payload_bpw", "scale",
                         "scale_metadata_bits_per_matrix", "nmse", "encode_seconds"))
        for (layer, projection), (values, *_) in split.items():
            calibration = hadamard_rows(screen[layer, projection][0])
            target = hadamard_rows(values[[0]])
            base = np.sqrt(np.mean(np.square(calibration, dtype=np.float64)))
            scales = (base * np.array((0.8, 1., 1.25, 1.5))).astype(np.float16).astype(np.float32)

            def lattice_fit(rows, scale, fn):
                return fn(rows.reshape(-1, 8), lattice, scale).reshape(rows.shape)

            def trellis_fit(rows, scale, width):
                return np.stack([trellis_row(row / scale, width) * scale for row in rows])

            methods = (
                ("e8_shaped_rht64", 2, lambda x, s: lattice_fit(x, s, lattice_quantize)),
                ("e8_residual_rht64", 4, lambda x, s: lattice_fit(x, s, e8_residual)),
                ("trellis_1mad_rht64", 3, lambda x, s: trellis_fit(x, s, 3)),
                ("trellis_1mad_rht64", 4, lambda x, s: trellis_fit(x, s, 4)),
            )
            energy = np.square(values[0], dtype=np.float64).sum()
            for method, rate, fit in methods:
                started = time.perf_counter()
                best = min(scales, key=lambda scale: np.square(
                    fit(calibration, scale) - calibration, dtype=np.float64).sum())
                reconstruction = hadamard_rows(fit(target, best), inverse=True)
                error = np.square(reconstruction - values[0], dtype=np.float64).sum() / energy
                writer.writerow((method, layer, projection, target.size, rate, best, 16, error,
                                 time.perf_counter() - started))
            handle.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("oracle", "subspaces", "vectors", "rotated", "all"),
                        default="all")
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--oracle-blocks", type=int, default=2)
    args = parser.parse_args()
    if not (1 <= args.blocks <= 48 and 1 <= args.oracle_blocks <= 48):
        parser.error("Need 1–48 blocks per matrix")
    args.out.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    (args.out / "sampling.json").write_text(json.dumps(dict(
        checkpoint_sha256=checksum, layers=[0, 15, 29], rng_seed=20261124,
        oracle_blocks=args.oracle_blocks, subspace_blocks=args.blocks, spans=2), indent=2) + "\n")
    split = samples(Weights(args.checkpoint), (0, 15, 29), 2, np.random.default_rng(20261124))
    if args.phase in ("oracle", "all"):
        oracle_metrics(args.out, split["holdout"], args.oracle_blocks)
    if args.phase in ("subspaces", "all"):
        projection_metrics(args.out, split["holdout"], args.blocks)
    if args.phase in ("vectors", "all"):
        other_metrics(args.out, split["screen"], split["holdout"])
    if args.phase in ("rotated", "all"):
        rotated_metrics(args.out, split["screen"], split["holdout"])


if __name__ == "__main__":
    main()
