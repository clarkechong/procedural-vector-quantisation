"""Exhaustive four-weight, eight-bit computational codebook experiment.

Run from the repository root with ``python -m experiments.synthesis.miniature``.
The grammar and its finite dictionaries are deliberately fixed in this file.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.cluster.vq import kmeans2

from experiments.generator_algebra import bitvector
from experiments.generator_algebra.dictionary import hash32
from experiments.encoding.seedlm import lfsr
from experiments.manifests.weights import PROJECTIONS, Weights
from experiments.synthesis import grammar


OPS = ("ADD", "SUB", "MUL", "AND", "OR", "XOR", "NOT", "SHL", "SHR", "ROTL", "POPCOUNT")
CONSTANTS = (0, 1, 2, 3, 15, 255)
LAYERS = (0, 15, 29)
SEED = 20261126
TILES = 4096
WIDTH = 4


def sample(checkpoint):
    """Equal counts from 21 matrices, with independent rows for each split."""
    rng = np.random.default_rng(SEED)
    weights = Weights(checkpoint)
    matrices = [(layer, projection) for layer in LAYERS for projection in PROJECTIONS]
    groups = {"calibration": [], "heldout": []}
    manifest = []
    for index, (layer, projection) in enumerate(matrices):
        matrix = weights.get(layer, projection)
        counts = [TILES // len(matrices) + (index < TILES % len(matrices)),
                  TILES // len(matrices) + (index < TILES % len(matrices))]
        row_partition = rng.permutation(matrix.shape[0])
        row_halves = np.array_split(row_partition, 2)
        for split, count, available in (("calibration", counts[0], row_halves[0]),
                                        ("heldout", counts[1], row_halves[1])):
            chosen = rng.choice(available, count, replace=True)
            offsets = rng.integers(0, matrix.shape[1] // WIDTH, size=count) * WIDTH
            groups[split].append(matrix[chosen[:, None], offsets[:, None] + np.arange(WIDTH)])
            manifest.extend(dict(split=split, layer=layer, projection=projection,
                                 row=int(row), start=int(start))
                            for row, start in zip(chosen, offsets))
    return {key: np.concatenate(value).astype(np.float32) for key, value in groups.items()}, manifest


def candidate_programs():
    options = tuple(tuple(grammar.choices(length, OPS, CONSTANTS)) for length in (0, 1))
    for instruction in options[0]:
        yield (instruction,)
    for prefix in options[0]:
        yield from grammar.extensions((prefix,), options)


def code(program):
    return np.ascontiguousarray(bitvector.codebook(program, 8, WIDTH))


def nearest_errors(codebooks, targets, scale, device, dtype=torch.float32):
    """Exact codeword search; the torch dot product only changes evaluation order."""
    books = torch.as_tensor(codebooks.astype(np.float64) * scale, device=device, dtype=dtype)
    points = torch.as_tensor(targets, device=device, dtype=dtype)
    norms = (books * books).sum(-1)
    dot = torch.bmm(books, points.T.expand(len(books), -1, -1))
    cost = norms[:, :, None] - 2 * dot
    winners = cost.argmin(1)
    selected = torch.gather(books, 1, winners[:, :, None].expand(-1, -1, WIDTH))
    residual = selected - points[None, :, :]
    return (residual.double().square().sum((1, 2))).cpu().numpy()


def algebra(book):
    """Properties of all 256 codewords, independent of sampled weights."""
    values = book.astype(np.float64)
    centered = values - values.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    spectrum = singular * singular
    proportions = spectrum / spectrum.sum() if spectrum.sum() else spectrum
    effective_rank = float(np.exp(-np.sum(proportions[proportions > 0] *
                                          np.log(proportions[proportions > 0]))))
    # Real-valued multilinear expansion over the Boolean seed cube.
    coefficients = values.copy()
    for bit in range(8):
        indices = np.arange(256)
        high = indices[(indices & (1 << bit)) != 0]
        coefficients[high] -= coefficients[high ^ (1 << bit)]
    active = np.any(np.abs(coefficients) > 1e-9, axis=1)
    degree = max((int(index).bit_count() for index in np.flatnonzero(active)), default=0)
    locality = float(np.mean(np.abs(values[np.arange(256) ^ 1] - values)))
    return {"effective_rank": effective_rank, "polynomial_degree": degree,
            "seed_bit0_mean_abs_change": locality}


def score_one(book, targets, scale, device):
    return float(nearest_errors(book[None], targets, scale, device)[0] /
                 np.square(targets.astype(np.float64)).sum())


def bf16_code(program):
    seeds = np.arange(256, dtype=np.uint32)[:, None]
    indices = np.arange(4, dtype=np.uint32)
    values = {"s": seeds, "i": indices}
    for step, instruction in enumerate(program):
        values[f"r{step}"] = bitvector.vector_instruction(instruction, values)
    raw = np.broadcast_to(values[f"r{len(program)-1}"], (256, 4))
    bits = (raw & np.uint32(65535)).astype(np.uint32)
    if np.any(((bits >> 7) & 255) == 255):
        return None
    return np.ascontiguousarray((bits << 16).view(np.float32))


def direct_bf16_search(cal, hold, device):
    """Same grammar with final low 16 bits read directly as BF16."""
    seen = set()
    rows = []
    pending = []
    books = []
    energy = float(np.square(cal.astype(np.float64)).sum())

    def flush():
        if not books:
            return
        errors = nearest_errors(np.stack(books), cal, 1, device, torch.float64)
        for row, error in zip(pending, errors):
            row["calibration_nmse"] = float(error / energy)
        rows.extend(pending)
        pending.clear()
        books.clear()

    for program in candidate_programs():
        book = bf16_code(program)
        if book is None:
            continue
        key = hashlib.sha256(book.tobytes()).digest()
        if key in seen:
            continue
        seen.add(key)
        pending.append({"program": bitvector.expression(program), "instructions": len(program)})
        books.append(book)
        if len(books) == 16:
            flush()
    flush()
    rows.sort(key=lambda row: (row["calibration_nmse"], row["instructions"]))
    for row in rows[:1]:
        row["heldout_nmse"] = float(nearest_errors(
            bf16_code(next(p for p in candidate_programs()
                           if bitvector.expression(p) == row["program"]))[None],
            hold, 1, device, torch.float64)[0] /
            np.square(hold.astype(np.float64)).sum())
    return rows


def scalar_baseline(cal, hold):
    # Four independently addressable scalar symbols consume exactly 2 bits/weight.
    levels = np.quantile(cal.ravel(), [0.125, 0.375, 0.625, 0.875]).astype(np.float32)
    for _ in range(100):
        ids = np.argmin(np.square(cal.ravel()[:, None] - levels[None, :]), axis=1)
        updated = np.array([cal.ravel()[ids == i].mean() if np.any(ids == i) else levels[i]
                            for i in range(4)], dtype=np.float32)
        if np.array_equal(updated, levels):
            break
        levels = updated
    levels = levels.astype(np.float16).astype(np.float32)
    values = levels[np.argmin(np.square(hold[:, :, None] - levels[None, None, :]), axis=2)]
    return float(np.square(hold.astype(np.float64) - values).sum() /
                 np.square(hold.astype(np.float64)).sum()), levels


def controls(cal, hold, scale, device):
    rng = np.random.default_rng(SEED + 1)
    result = {}
    scalar, levels = scalar_baseline(cal, hold)
    result["lloyd_max_scalar"] = {"heldout_nmse": scalar, "shared_bytes": 8,
                                  "levels": levels.tolist()}
    std = float(cal.std(dtype=np.float64))
    mean = float(cal.mean(dtype=np.float64))
    gaussian = rng.normal(mean, std, (256, 4)).astype(np.float32)
    gaussian = gaussian.astype(np.float16).astype(np.float32)
    result["gaussian_random_vq"] = {"heldout_nmse": score_one(gaussian, hold, 1, device),
                                     "shared_bytes": 2048}
    # Multiple starts avoid attributing an unlucky K-means initialization to VQ.
    trained = []
    for start in range(3):
        centers, _ = kmeans2(cal, cal[rng.choice(len(cal), 256, replace=False)],
                             iter=50, minit="matrix", missing="warn")
        trained.append((score_one(centers, cal, 1, device), centers))
    centers = min(trained, key=lambda pair: pair[0])[1].astype(np.float16).astype(np.float32)
    result["trained_vq"] = {"heldout_nmse": score_one(centers, hold, 1, device),
                             "shared_bytes": 2048}
    seeds = np.arange(256, dtype=np.uint32)
    words = hash32(seeds[:, None] * np.uint32(4) + np.arange(4, dtype=np.uint32))
    prng = (words & 255).astype(np.uint8).view(np.int8)
    result["prng_signed_byte"] = {"heldout_nmse": score_one(prng, hold, scale, device),
                                   "shared_bytes": 4}
    # A manually structured 8-bit latent: 4-bit LFSR seed plus 4-bit
    # power-of-two scale. This is SeedLM-inspired, not its published format.
    patterns = lfsr(np.arange(16, dtype=np.uint32), 4, 4)
    exponent = (np.arange(16, dtype=np.int32) - 8)
    structured = (patterns[:, None, :] * np.exp2(exponent)[None, :, None]).reshape(256, 4)
    result["lfsr_seed_exponent"] = {
        "heldout_nmse": score_one(structured, hold, float(cal.std()), device),
        "shared_bytes": 4}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/smollm2-135m.safetensors")
    parser.add_argument("--out", default="results/e01-miniature-codebook")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    groups, manifest = sample(args.checkpoint)
    cal, hold = groups["calibration"], groups["heldout"]
    # One global FP32 scale, determined before synthesis and counted as 4 shared bytes.
    scale = float(np.sqrt(np.mean(np.square(cal.astype(np.float64)))) / 32)
    rng = np.random.default_rng(SEED + 2)
    shuffled = rng.permutation(cal.ravel()).reshape(cal.shape)
    gaussian = rng.normal(cal.mean(), cal.std(), cal.shape).astype(np.float32)
    planted_program = (bitvector.Instruction("XOR", ("s", "i")),)
    planted_book = (code(planted_program).astype(np.float64) * scale).astype(np.float32)
    planted = planted_book[rng.integers(0, 256, len(cal))]
    targets = {"original": cal, "shuffled": shuffled, "gaussian": gaussian, "planted": planted}
    energies = {name: float(np.square(value.astype(np.float64)).sum()) for name, value in targets.items()}
    rows = []
    books = []
    pending = []
    signatures = {}
    duplicate = 0
    aliases = []

    def flush():
        if not books:
            return
        batch = np.stack(books)
        for name, values in targets.items():
            for row, error in zip(pending, nearest_errors(batch, values, scale, args.device)):
                row[name + "_nmse"] = float(error / energies[name])
        rows.extend(pending)
        books.clear()
        pending.clear()

    for program in candidate_programs():
        values = code(program)
        key = hashlib.sha256(values.tobytes()).hexdigest()
        if key in signatures:
            duplicate += 1
            aliases.append((bitvector.expression(program), signatures[key]))
            continue
        signatures[key] = len(signatures)
        pending.append({"id": len(signatures) - 1, "instructions": len(program),
                        "program": bitvector.expression(program),
                        "collisions": 256 - len(np.unique(values, axis=0)),
                        "affine_rank": int(np.linalg.matrix_rank(values.astype(np.float64) - values[0])),
                        "dependency_depth": grammar.cost(program, WIDTH)["dependency_depth"],
                        **algebra(values)})
        books.append(values)
        if len(books) == 32:
            flush()
            if len(rows) % 1024 == 0:
                print(f"scored {len(rows)} unique codebooks in {time.monotonic()-start:.1f}s", flush=True)
    flush()
    rows.sort(key=lambda row: (row["original_nmse"], row["instructions"], row["id"]))
    with (out / "ranked.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (out / "equivalent_programs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("program", "representative_id"))
        writer.writerows(aliases)
    winners = {}
    for budget in (1, 2):
        row = next(row for row in rows if row["instructions"] <= budget)
        winners[str(budget)] = {**row, "heldout_nmse": score_one(
            code_from_expression(row["program"]), hold, scale, args.device)}
    shuffled_hold = rng.permutation(hold.ravel()).reshape(hold.shape)
    gaussian_hold = rng.normal(cal.mean(), cal.std(), hold.shape).astype(np.float32)
    control_winners = {}
    for name, held in (("shuffled", shuffled_hold), ("gaussian", gaussian_hold)):
        row = min(rows, key=lambda entry: entry[name + "_nmse"])
        control_winners[name] = {**row, "heldout_nmse": score_one(
            code_from_expression(row["program"]), held, scale, args.device)}
    planted_winner = min(rows, key=lambda row: row["planted_nmse"])
    planted_exact = next(row for row in rows if row["program"] == bitvector.expression(planted_program))
    baselines = controls(cal, hold, scale, args.device)
    bf16_rows = direct_bf16_search(cal, hold, args.device)
    with (out / "direct_bf16_ranked.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ("program", "instructions", "calibration_nmse", "heldout_nmse"))
        writer.writeheader()
        for row in bf16_rows:
            writer.writerow(row)
    result = {"grammar": {"operations": OPS, "constants": CONSTANTS, "seed_bits": 8,
                           "tile_width": WIDTH, "max_instructions": 2,
                           "output": "signed low byte of final uint32, times one global FP32 scale"},
              "sampling": {"layers": LAYERS, "projections": list(PROJECTIONS),
                           "calibration_tiles": len(cal), "heldout_tiles": len(hold), "seed": SEED},
              "global_scale": scale, "enumerated_programs": len(rows) + duplicate,
              "unique_codebooks": len(rows), "equivalent_programs": duplicate,
              "winners": winners, "planted_winner": planted_winner,
              "planted_program_score": planted_exact["planted_nmse"],
              "control_winners": control_winners,
              "direct_bf16": {"valid_unique_codebooks": len(bf16_rows),
                              "winner": bf16_rows[0] if bf16_rows else None},
              "baselines": baselines, "elapsed_seconds": time.monotonic() - start,
              "storage": {"tile_payload_bits": 8, "global_scale_bytes": 4,
                          "program_bytecode_bytes": {"1": 3, "2": 6},
                          "program_bytecode_rule": "opcode, operand A, operand B per instruction; unary B is zero; dictionary constants are decoder-defined"}}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rate_rows = []
    for budget, winner in winners.items():
        overhead = 4 + int(budget) * 3
        rate_rows.append((f"synthesized_{budget}_op", winner["heldout_nmse"],
                          2 + 8 * overhead / (WIDTH * len(hold)), overhead))
    for name, baseline in baselines.items():
        rate_rows.append((name, baseline["heldout_nmse"],
                          2 + 8 * baseline["shared_bytes"] / (WIDTH * len(hold)),
                          baseline["shared_bytes"]))
    if bf16_rows:
        direct = bf16_rows[0]
        overhead = 3 * direct["instructions"]
        rate_rows.append(("direct_bf16", direct["heldout_nmse"],
                          2 + 8 * overhead / (WIDTH * len(hold)), overhead))
    with (out / "rate_distortion.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "heldout_nmse", "effective_bpw_on_4096_tiles", "shared_bytes"))
        writer.writerows(rate_rows)
    (out / "sample_manifest.json").write_text(json.dumps(manifest) + "\n")
    print(json.dumps({"winners": winners, "planted_winner": planted_winner,
                      "baselines": baselines, "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


def code_from_expression(expression):
    """Only expressions written by our own enumerator are accepted."""
    for program in candidate_programs():
        if bitvector.expression(program) == expression:
            return code(program)
    raise ValueError(expression)


if __name__ == "__main__":
    main()
