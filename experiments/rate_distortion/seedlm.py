"""Reconstruct published SeedLM configurations and screen seeded generator families."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from experiments.encoding.pursuit import pursuit, quantize, seeded_basis
from experiments.encoding.seedlm import (
    Config, FAMILIES, PUBLISHED, TAPS, check_reconstruction, search, seed_fields, seed_ids,
)
from experiments.generator_algebra.dictionary import dictionary
from experiments.manifests.sampling import samples
from experiments.manifests.weights import PROJECTIONS, Weights


TARGETS = (1, 2, 3, 4)
WIDTHS = (4, 8, 12, 16)
BLOCKS = (4, 8, 12, 16, 32, 64)
DATASETS = ("pretrained", "global_shuffle", "within_block_shuffle", "gaussian")
FIELDS = ("experiment", "block", "latent", "seed_bits", "coefficient_bits", "family",
          "bits_per_weight", "fitting", "rule", "candidates", "layer", "projection", "dataset",
          "blocks", "sse", "energy", "nmse")
DECODE_FIELDS = ("family", "block", "latent", "seed_bits", "coefficient_bits",
                 "bits_per_weight", "basis_values_per_weight", "generator_steps_per_weight",
                 "approx_integer_ops_per_weight", "multiply_accumulates_per_weight")


def block_samples(original, shuffled, gaussian, width, rng):
    a, b, c = (values.reshape(-1, width) for values in (original, shuffled, gaussian))
    return np.stack((a, b, rng.permuted(a, axis=1), c))


def make_row(experiment, config, fitting, rule, count, layer, projection, dataset, errors, blocks):
    energy = float(np.sum(np.square(blocks, dtype=np.float64)))
    total = float(np.sum(errors, dtype=np.float64))
    return {
        "experiment": experiment, **asdict(config), "bits_per_weight":
            config.rate if rule != "float" else (config.seed_bits + 32 * config.latent) / config.block,
        "fitting": fitting, "rule": rule, "candidates": count, "layer": layer,
        "projection": projection, "dataset": dataset, "blocks": len(blocks),
        "sse": total, "energy": energy, "nmse": total / energy if energy else 0.0,
    }


def evaluate(writer, split, experiment, config, rules, fitting="joint", prefixes=()):
    print(f"{experiment}: {config} {fitting}", flush=True)
    started = time.perf_counter()
    for (layer, projection), values in split.items():
        rng = np.random.default_rng([layer, list(PROJECTIONS).index(projection), config.block])
        cases = block_samples(*values, config.block, rng)
        flat = cases.reshape(-1, config.block)
        total_candidates = len(seed_ids(config))
        for count, rule, result in search(config, flat, rules, fitting, prefixes):
            if rule != "float" and count == total_candidates:
                check_reconstruction(config, flat, result)
            for dataset, blocks, errors in zip(DATASETS, cases, result[0].reshape(cases.shape[:2])):
                writer.writerow(make_row(experiment, config, fitting, rule, count, layer,
                                         projection, dataset, errors, blocks))
    return time.perf_counter() - started


def screen(writer, split, limit):
    """Screen feasible payloads on training rows using a bounded seed prefix."""
    configs = []
    for block, seed, latent, bits in itertools.product(BLOCKS, WIDTHS, range(1, 9), (2, 3, 4, 8)):
        config = Config(block, latent, seed, bits)
        if latent < block and 0.75 <= config.rate <= 4:
            configs.append(config)
    for config in PUBLISHED:
        configs.extend(replace(config, family=family) for family in FAMILIES
                       if family not in ("lfsr", "independent_lfsr"))
    print(f"Screening {len(configs)} configurations with at most {limit} seeds each", flush=True)
    results = []
    for config in configs:
        # Up to four blocks per matrix, weighted equally in the screening objective.
        per_matrix = min(4, min(values[0].size for values in split.values()) // config.block)
        blocks = np.concatenate([
            values[0].ravel()[:per_matrix * config.block].reshape(-1, config.block)
            for values in split.values()
        ])
        energy = np.sum(np.square(blocks, dtype=np.float64), axis=1).reshape(
            len(split), per_matrix
        ).sum(axis=1)
        count, rule, result = next(search(config, blocks, ("range",), limit=limit))
        errors = result[0].reshape(len(split), per_matrix).sum(axis=1)
        ratios = np.divide(errors, energy, out=np.zeros_like(energy), where=energy > 0)
        nmse = float(ratios.mean())
        results.append((nmse, config))
        writer.writerow({**asdict(config), "bits_per_weight": config.rate, "candidates": count,
                         "nmse": nmse})
    return {target: min((row for row in results if row[1].rate <= target),
                        key=lambda row: row[0])[1] for target in TARGETS}


def baseline_rows(writer, split):
    for (layer, projection), values in split.items():
        original = values[0]
        for granularity in (8, 12, 64):
            blocks = original.reshape(-1, granularity)
            energy = float(np.sum(np.square(blocks, dtype=np.float64)))
            geometry = (1, granularity)
            atoms, _ = dictionary(geometry, 16)
            methods = [(f"old_pursuit_{count}", 24 * count, error)
                       for count, error in pursuit(blocks, atoms, 4 * granularity // 24)]
            methods.extend((f"old_seeded_basis_{count}", 8 + 16 * count, error)
                           for count, error in seeded_basis(blocks, 16, (1, 2, 3)))
            for width in (1, 2, 3, 4):
                for method, nonuniform in (("uniform", False), ("lloyd_max", True)):
                    size, errors = quantize(blocks, width, nonuniform)
                    methods.append((f"{method}_{width}", size, errors))
            for method, bits, error in methods:
                total = float(error.sum(dtype=np.float64))
                writer.writerow((layer, projection, granularity, method, bits / granularity,
                                 total, energy, total / energy))


def operation_counts(config):
    """Count scalar basis transforms as a proxy for online decode work."""
    family = config.family
    if family == "independent_lfsr":
        costs = [3 * len(TAPS[width]) for width, _ in seed_fields(config)]
        work = sum(costs)
    elif family == "lfsr":
        work = 3 * len(TAPS[config.seed_bits]) * config.latent
    elif family == "mixed":
        # Counts include direct Walsh signs and the reference's full LFSR stream.
        work = (3 * len(TAPS[config.seed_bits]) + 4) * config.latent + 9
    else:
        work = {"xorshift": 6 * config.latent + 8 / config.block,
                "counter": 8 * config.latent,
                "splitmix": 9 * config.latent,
                "hadamard": 4 * config.latent + 9}[family]
    values = asdict(config)
    values.update(bits_per_weight=config.rate, basis_values_per_weight=config.latent,
                  generator_steps_per_weight=2 * config.latent if family == "mixed" else config.latent,
                  approx_integer_ops_per_weight=work, multiply_accumulates_per_weight=config.latent)
    return values


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = Weights(args.checkpoint)
    provenance = {}
    for path in sorted({entry[0] for entry in checkpoint.tensors.values()}):
        with path.open("rb") as handle:
            provenance[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    splits = samples(checkpoint, args.layers, args.spans, np.random.default_rng(args.seed))
    timings = []
    tested = set()
    finalists = {}
    with (args.out / "measurements.csv").open("w", newline="") as data:
        writer = csv.DictWriter(data, fieldnames=FIELDS)
        writer.writeheader()

        def test(label, config, rules, fitting="joint", prefixes=()):
            duration = evaluate(writer, splits["holdout"], label, config, rules, fitting, prefixes)
            timings.append({"experiment": label, **asdict(config), "fitting": fitting,
                            "rules": list(rules), "seconds": duration})
            tested.add(config)
            data.flush()

        # Phase A always runs before further design changes.
        for config in PUBLISHED:
            independent = replace(config, family="independent_lfsr")
            prefixes = (1, 15, 255, 4095, len(seed_ids(independent)))
            test("published", config, ("paper", "range", "float"), prefixes=prefixes)
        if args.phase != "reference":
            for config in PUBLISHED:
                test("fitting", config, ("range", "paper"), fitting="greedy")
                test("quantization_aware", config, ("refine",))
                for family in FAMILIES:
                    if family != "lfsr":
                        variant = replace(config, family=family)
                        test("matched_family", variant, ("range", "paper"))
        if args.phase == "all":
            with (args.out / "screen.csv").open("w", newline="") as handle:
                screen_writer = csv.DictWriter(handle, fieldnames=[
                    "block", "latent", "seed_bits", "coefficient_bits", "family",
                    "bits_per_weight", "candidates", "nmse",
                ])
                screen_writer.writeheader()
                finalists = screen(screen_writer, splits["screen"], args.screen_limit)
            for config in dict.fromkeys(finalists.values()):
                test("screen_finalist", config, ("range", "refine"))

    with (args.out / "scalar_and_pilot_baselines.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("layer", "projection", "block", "method", "bits_per_weight",
                         "sse", "energy", "nmse"))
        baseline_rows(writer, splits["holdout"])
    with (args.out / "decode_cost.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECODE_FIELDS)
        writer.writeheader()
        writer.writerows(operation_counts(config) for config in
                         sorted(tested, key=lambda c: (c.rate, c.block, c.family)))
    metadata = {**vars(args), "checkpoint_sha256": provenance, "numpy": np.__version__,
                "finalists": {target: asdict(config) for target, config in finalists.items()},
                "encoding_seconds": timings}
    (args.out / "config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    from experiments.rate_distortion.seedlm_report import report

    report(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 15, 29])
    parser.add_argument("--spans", type=int, default=2, help="192-weight spans per matrix and split")
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--phase", choices=("reference", "ablations", "all"), default="all")
    parser.add_argument("--screen-limit", type=int, default=1024)
    args = parser.parse_args()
    if min(args.spans, args.screen_limit) < 1 or len(set(args.layers)) != len(args.layers):
        parser.error("Provide unique layers and positive spans and screening limit.")
    run(args)


if __name__ == "__main__":
    main()
