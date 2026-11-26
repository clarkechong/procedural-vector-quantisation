"""Run bounded uint32 generator synthesis with pretraining-weight holdouts."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import heapq
from itertools import islice
import json
from pathlib import Path
import platform
import time

import numpy as np

from experiments.encoding import synthesized
from experiments.encoding.seedlm import Config, TAPS, check_reconstruction, search, splitmix64
from experiments.generator_algebra import bitvector
from experiments.generator_algebra.dictionary import hash32
from experiments.manifests.sampling import samples as previous_samples
from experiments.manifests.weights import PROJECTIONS, Weights
from experiments.synthesis import grammar


WIDTH = 8
SPLIT_SEED = 810642
ENUMERATION_SEED = 901416
RATES = (6, 8, 12, 16)
LAYERS = (0, 15, 29)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0])
        writer.writeheader()
        writer.writerows(rows)


def sample_weights(checkpoint):
    """Reserve different complete matrix rows for every split."""
    rng = np.random.default_rng(SPLIT_SEED)
    weights = Weights(checkpoint)
    previous_rows = {}
    previous_samples(weights, LAYERS, 2, np.random.default_rng(20261124), row_manifest=previous_rows)
    samples = defaultdict(list)
    manifest = []
    for layer in LAYERS:
        for projection in PROJECTIONS:
            matrix = weights.get(layer, projection)
            eligible = np.setdiff1d(np.arange(matrix.shape[0]), previous_rows[layer, projection])
            rows = rng.choice(eligible, 45, replace=False)
            for split, count in (("calibration", 32), ("screen", 1),
                                 ("validation", 4), ("holdout", 8)):
                for row in rows[:count]:
                    blocks = rng.choice(matrix.shape[1] // WIDTH, 16, replace=False) * WIDTH
                    manifest.append(dict(split=split, layer=layer, projection=projection,
                                         row=int(row), block_starts=blocks.tolist()))
                    samples[split].extend(matrix[row, blocks[:, None] + np.arange(WIDTH)])
                rows = rows[count:]
    return {name: np.asarray(blocks) for name, blocks in samples.items()}, manifest


def squared_error(blocks, reconstruction):
    return np.sum(np.square(blocks.astype(np.float64) - reconstruction), axis=1)


def nmse(blocks, errors):
    return float(np.sum(errors) / np.sum(np.square(blocks, dtype=np.float64)))


def find_nearest(vectors, blocks):
    parts = [synthesized.exact_search(vectors, chunk) for chunk in np.array_split(
        blocks, max(1, (len(blocks) + 95) // 96)
    )]
    return tuple(np.concatenate(fields) for fields in zip(*parts))


def depends_on_both(program):
    inputs = {"s": 1, "i": 2}
    for step, instruction in enumerate(program):
        flags = 0
        for arg in instruction.args:
            flags |= inputs.get(arg, 0)
        inputs[f"r{step}"] = flags
    return flags == 3


def budgeted_three(options, leaders, limit, counters):
    """Try promising two-op DAG prefixes, then a random permutation of other prefixes."""
    rng = np.random.default_rng(ENUMERATION_SEED)
    count = len(options[0]) * len(options[1])
    quota = limit // 3
    visited = {}
    for prefix in leaders:
        if counters["guided"] >= quota:
            break
        allowed = quota - counters["guided"]
        visited[prefix] = 0
        for program in islice(grammar.extensions(prefix, options), allowed):
            visited[prefix] += 1
            counters["guided"] += 1
            yield program
        if visited[prefix] < allowed:
            visited[prefix] = None
    for flat in rng.permutation(count):
        if sum(counters.values()) >= limit:
            break
        first, second = divmod(int(flat), len(options[1]))
        prefix = (options[0][first], options[1][second])
        skip = visited.get(prefix, 0)
        if skip is None:
            continue
        for program in islice(grammar.extensions(prefix, options), skip, None):
            if sum(counters.values()) >= limit:
                break
            counters["random"] += 1
            yield program


def run_enumeration(args, samples, out):
    options = tuple(tuple(grammar.choices(n, args.ops, args.constants)) for n in range(args.max_instructions))
    screen = grammar.SemanticScreen(args.seed_bits, WIDTH)
    # A fixed two-block subset per projection from the screening rows.
    training = samples["screen"].reshape(len(LAYERS) * len(PROJECTIONS), 16, WIDTH)[:, :2].reshape(-1, WIDTH)
    energy = np.sum(training.astype(np.float64) ** 2)
    timings, audit_rows, finalists = [], [], {}
    two_op_leaders = []
    scores_path = out / "screen.csv"
    with scores_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["program_id", "program", "screen_nmse", "instructions",
                         "dependency_depth", "peak_live_values", "cost_per_block"])
        for length in range(1, args.max_instructions + 1):
            budget_counters = Counter()
            if length == 3 and args.three_limit:
                programs = budgeted_three(options, two_op_leaders, args.three_limit, budget_counters)
            else:
                programs = (program for prefix in grammar.prefixes(length, options)
                            for program in grammar.extensions(prefix, options))
            before = screen.audit.copy()
            stats = Counter()
            top = []
            start = time.perf_counter()
            search_seconds = 0.0
            for program in programs:
                stats["live_dags"] += 1
                if not depends_on_both(program):
                    stats["syntactically_independent"] += 1
                    continue
                if not screen.accept(program):
                    continue
                stats["scored"] += 1
                code = screen.materialize(program)
                tick = time.perf_counter()
                errors, *_ = synthesized.exact_search(code, training)
                search_seconds += time.perf_counter() - tick
                value = float(np.sum(errors) / energy)
                name = f"{length}-{stats['scored']}"
                costs = grammar.cost(program, WIDTH)
                writer.writerow((name, bitvector.expression(program), value, *costs.values()))
                entry = (-value, -stats["scored"], name, program)
                if len(top) < args.finalists:
                    heapq.heappush(top, entry)
                elif entry > top[0]:
                    heapq.heapreplace(top, entry)
            finalists[length] = [(name, program, -score) for score, _, name, program in sorted(top, reverse=True)]
            if length == 2:
                two_op_leaders = [program for _, program, _ in finalists[length]]
            stats.update(screen.audit - before)
            seconds = time.perf_counter() - start
            audit_rows.append(dict(
                instructions=length, exhaustive=not (length == 3 and args.three_limit),
                raw_prefixes=int(np.prod([len(x) for x in options[:length - 1]], dtype=np.int64)),
                live_dags=stats["live_dags"], guided=budget_counters["guided"],
                random=budget_counters["random"],
                syntactically_independent=stats["syntactically_independent"],
                degenerate=stats["degenerate"], equivalent=stats["equivalent"],
                collision_checks=stats["collision_checks"], distinct_collisions=stats["distinct_collisions"],
                distinct_scored=stats["scored"], total_seconds=seconds,
                exact_search_seconds=search_seconds,
            ))
            print(audit_rows[-1], flush=True)
            handle.flush()
    write_csv(out / "enumeration.csv", audit_rows)

    # Stratified workload estimate from uniformly sampled structural prefixes.
    if len(options) >= 3:
        rng = np.random.default_rng(ENUMERATION_SEED + 1)
        population = len(options[0]) * len(options[1])
        for length in (2, 3):
            prefix_count = len(options[0]) if length == 2 else population
            selected = rng.choice(prefix_count, min(512, prefix_count), replace=False)
            start = time.perf_counter()
            counts = []
            for index in selected:
                if length == 2:
                    prefix = (options[0][index],)
                else:
                    first, second = divmod(int(index), len(options[1]))
                    prefix = (options[0][first], options[1][second])
                counts.append(sum(1 for _ in grammar.extensions(prefix, options)))
            timings.append(dict(instructions=length, prefixes=prefix_count,
                                sampled_prefixes=len(selected), sampled_live_dags=sum(counts),
                                estimated_live_dags=prefix_count * np.mean(counts),
                                estimate_standard_error=(prefix_count * np.std(counts, ddof=1)
                                * np.sqrt((prefix_count - len(counts)) / (prefix_count - 1) / len(counts))
                                if len(counts) > 1 else 0),
                                traversal_seconds=time.perf_counter() - start))
        write_csv(out / "structural_scaling.csv", timings)
    return finalists


def direct_family(family, seed_bits):
    seeds = np.arange(1 << seed_bits, dtype=np.uint32)
    if family == "LFSR":
        tap_mask = sum(1 << tap for tap in TAPS[seed_bits])
        state = seeds.copy()
        columns = []
        for _ in range(WIDTH):
            feedback = (np.bitwise_count(state & tap_mask).astype(np.uint32) & 1) << (seed_bits - 1)
            state = (state >> 1) | feedback
            columns.append((state >> (seed_bits - 8)) if seed_bits >= 8 else (state << (8 - seed_bits)))
        return np.stack(columns, axis=1).astype(np.uint8).view(np.int8)
    if family == "Xorshift":
        state = hash32(seeds)
        columns = []
        for _ in range(WIDTH):
            state ^= state << 13
            state ^= state >> 17
            state ^= state << 5
            columns.append(state >> 24)
        return np.stack(columns, axis=1).astype(np.uint8).view(np.int8)
    if family != "SplitMix":
        raise ValueError(f"Unknown direct family {family}")
    counters = (seeds[:, None].astype(np.uint64) << 32) | np.arange(WIDTH, dtype=np.uint64)
    words = splitmix64(counters + np.uint64(0x9E3779B97F4A7C15))
    return (words >> 56).astype(np.uint8).view(np.int8)


def lloyd_max(training, targets, payload_bits):
    """Fit per-matrix FP16 scalar levels, with a fixed bit allocation across positions."""
    count = len(LAYERS) * len(PROJECTIONS)
    train_shards = np.split(training, count)
    target_shards = np.split(targets, count)
    depth, extra = divmod(payload_bits, WIDTH)
    reconstructed = []
    for train, target in zip(train_shards, target_shards):
        row = np.empty_like(target)
        for bits in sorted({depth + (index < extra) for index in range(WIDTH)}):
            indices = [index for index in range(WIDTH) if depth + (index < extra) == bits]
            values = train[:, indices].ravel()
            levels = 1 << bits
            centers = np.quantile(values, (np.arange(levels) + 0.5) / levels)
            for _ in range(32):
                labels = np.searchsorted((centers[1:] + centers[:-1]) / 2, values)
                counts = np.bincount(labels, minlength=levels)
                sums = np.bincount(labels, weights=values, minlength=levels)
                np.divide(sums, counts, out=centers, where=counts > 0)
            centers = centers.astype(np.float16).astype(np.float32)
            labels = np.searchsorted((centers[1:] + centers[:-1]) / 2, target[:, indices])
            row[:, indices] = centers[labels]
        reconstructed.append(row)
    return np.concatenate(reconstructed), 2 * count * ((1 << depth) + ((2 << depth) if extra else 0))


def diagnostic_codebooks(seed_bits, training):
    rng = np.random.default_rng([5457, seed_bits])
    count = 1 << seed_bits
    gaussian = np.clip(np.rint(6 * rng.normal(size=(count, WIDTH))), -128, 127).astype(np.int8)
    if len(training) < count:
        return {"Gaussian codebook": gaussian}
    # Spherical Lloyd iterations fit directions, leaving magnitude to the exponent.
    data = training.copy()
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    data = (data[norm[:, 0] > 0] / norm[norm[:, 0] > 0])
    if len(data) < count:
        return {"Gaussian codebook": gaussian}
    centers = data[rng.choice(len(data), count, replace=False)].copy()
    for _ in range(8):
        totals = np.zeros_like(centers)
        for chunk in np.array_split(data, 32):
            labels = (chunk @ centers.T).argmax(axis=1)
            np.add.at(totals, labels, chunk)
        norm = np.linalg.norm(totals, axis=1, keepdims=True)
        np.divide(totals, norm, out=centers, where=norm > 0)
    trained = np.rint(16 * centers).astype(np.int8)
    return {"Gaussian codebook": gaussian, "Trained VQ": trained}


def diagnostics(vectors, seeds, errors):
    values = np.asarray(vectors, dtype=np.float64)
    centered = values - values.mean(axis=0)
    covariance = (centered.T @ centered) / len(values)
    eigen = np.maximum(np.linalg.eigvalsh(covariance), 0)
    proportions = eigen[eigen > 0] / max(eigen.sum(), 1e-30)
    rank = np.exp(-np.sum(proportions * np.log(proportions))) if len(proportions) else 0
    standard_deviations = np.sqrt(np.diag(covariance))
    denominators = np.outer(standard_deviations, standard_deviations)
    correlation = np.divide(covariance, denominators, out=np.zeros_like(covariance), where=denominators > 0)
    used, counts = np.unique(seeds, return_counts=True)
    histogram = np.histogram(values, bins=np.arange(-128, 129))[0].tolist()
    distances = np.quantile(np.sqrt(errors), [0, .05, .5, .95, 1]).tolist()
    return dict(unique_unscaled=len(np.unique(values, axis=0)), effective_rank=float(rank),
                mean_abs_offdiag_corr=float(np.abs(correlation[~np.eye(WIDTH, dtype=bool)]).mean()),
                used_seeds=len(used), most_used_seed_count=int(counts.max()), distance_quantiles=distances,
                covariance=covariance.tolist(), output_histogram=histogram,
                selected_seed_counts=dict(zip(used.tolist(), counts.tolist())))


def scaled_cardinality(vectors):
    """Count distinct scaled words. Scale by 256 to retain exact integer coordinates."""
    scaled = np.multiply(vectors[None, :, :].astype(np.int32), (1 << np.arange(16))[:, None, None])
    return len(np.unique(scaled.reshape(-1, WIDTH), axis=0))


def compare(args, samples, finalists, out, parameter_count):
    validation, holdout, calibration = (samples[name] for name in ("validation", "holdout", "calibration"))
    chosen = {}
    validation_rows, final_rows, projection_rows, baseline_validation = [], [], [], []
    details = {}
    programs = {}
    for length, candidates in finalists.items():
        best = None
        for name, program, screen_nmse in candidates:
            code = bitvector.codebook(program, args.seed_bits, WIDTH)
            errors, seeds, _, _ = find_nearest(code, validation)
            value = nmse(validation, errors)
            validation_rows.append(dict(program_id=name, program=bitvector.expression(program),
                                        screen_nmse=screen_nmse, validation_nmse=value,
                                        **grammar.cost(program, WIDTH)))
            details[f"validation/{name}"] = diagnostics(code, seeds, errors)
            key = (value, screen_nmse, name)
            if best is None or key < best:
                best = key
                chosen[length] = (name, program)
    write_csv(out / "validation.csv", validation_rows)
    (out / "selection.json").write_text(json.dumps({length: name for length, (name, _) in chosen.items()},
                                                   indent=2) + "\n")

    def record(name, bits, errors, *, vectors=None, seeds=None, overhead=0, program=None, note=""):
        cost = grammar.cost(program, WIDTH) if program else {
            "instructions": None, "dependency_depth": None, "peak_live_values": None, "cost_per_block": None
        }
        if vectors is not None:
            profile = diagnostics(vectors, seeds, errors)
            if bits == args.seed_bits + 4:
                profile["unique_scaled"] = scaled_cardinality(vectors)
            details[f"holdout/{bits}/{name}"] = profile
        else:
            profile = {}
        final_rows.append(dict(
            method=name, bits_per_weight=bits / WIDTH, bits_per_block=bits, nmse=nmse(holdout, errors),
            rmse=float(np.sqrt(np.mean(errors) / WIDTH)), unique_unscaled=profile.get("unique_unscaled"),
            unique_scaled=profile.get("unique_scaled"), effective_rank=profile.get("effective_rank"),
            used_seeds=profile.get("used_seeds"), shared_bytes=overhead,
            bpw_including_shared_on_sampled_matrices=bits / WIDTH + overhead * 8 / parameter_count,
            **cost, note=note,
        ))
        keys = [(layer, projection) for layer in LAYERS for projection in PROJECTIONS]
        for (layer, projection), actual, residual in zip(keys, np.split(holdout, len(keys)),
                                                         np.split(errors, len(keys))):
            projection_rows.append(dict(
                method=name, bits_per_weight=bits / WIDTH, layer=layer, projection=projection,
                blocks=len(residual), sse=float(np.sum(residual)),
                energy=float(np.sum(np.square(actual, dtype=np.float64))), nmse=nmse(actual, residual),
            ))

    def seeded_errors(config, blocks):
        errors = []
        for shard in np.array_split(blocks, max(1, (len(blocks) + 127) // 128)):
            _, _, result = next(search(config, shard, rules=("refine",), seed_batch=512))
            check_reconstruction(config, shard, result)
            errors.extend(result[0])
        return np.asarray(errors)

    def select_seedlm(bits):
        rows = []
        for basis_count in range(1, 5):
            for coefficient_bits in range(2, 9):
                seed_bits = bits - 4 - coefficient_bits * basis_count
                if seed_bits not in TAPS:
                    continue
                config = Config(WIDTH, basis_count, seed_bits, coefficient_bits)
                value = nmse(validation, seeded_errors(config, validation))
                rows.append((value, config))
        winner = min(rows, default=(None, None), key=lambda row: row[0])[1]
        baseline_validation.extend(dict(method="Refined SeedLM", bits_per_block=bits,
                                        latent=config.latent, seed_bits=config.seed_bits,
                                        coefficient_bits=config.coefficient_bits, output_shift=None,
                                        validation_nmse=value, selected=config == winner)
                                   for value, config in rows)
        return winner

    for seed_bits in sorted(set((*RATES, args.seed_bits))):
        bits = seed_bits + 4
        for length, (name, program) in chosen.items():
            vectors = bitvector.codebook(program, seed_bits, WIDTH)
            errors, seeds, exponents, reconstruction = find_nearest(vectors, holdout)
            payload = synthesized.pack(seeds, exponents, seed_bits)
            decoded = np.asarray(synthesized.decode(program, payload, seed_bits, WIDTH, count=len(holdout)))
            if not np.array_equal(decoded, reconstruction) or not np.array_equal(squared_error(holdout, decoded), errors):
                raise RuntimeError(f"Independent decode disagrees for {name} at {bits} bits")
            if seed_bits == args.seed_bits:
                (out / f"winner-{length}.bin").write_bytes(payload)
                programs[name] = dict(instructions=[asdict(x) for x in program], payload=f"winner-{length}.bin",
                                      payload_bytes=len(payload),
                                      sha256=hashlib.sha256(payload).hexdigest())
            # Fixed standalone descriptors use one opcode byte and up to two
            # tagged four-byte operands per instruction, and a length byte.
            overhead = 1 + sum(1 + 5 * len(instruction.args) for instruction in program)
            record(f"Synthesized {length}-op", bits, errors, vectors=vectors, seeds=seeds, overhead=overhead,
                   program=program, note=f"{name}; packed reconstruction verified")
        for family in ("LFSR", "Xorshift", "SplitMix"):
            if family == "LFSR" and seed_bits not in TAPS:
                continue
            vectors = direct_family(family, seed_bits)
            errors, seeds, *_ = find_nearest(vectors, holdout)
            costs = {"LFSR": "7 integer operations/weight", "Xorshift": "6 integer operations/weight + 8/block",
                     "SplitMix": "10 integer operations/weight + 1/block"}
            record(family, bits, errors, vectors=vectors, seeds=seeds, note=costs[family] + "; plus output and scale")
            if seed_bits == args.seed_bits:
                # A separate amplitude control lets conventional mixers adapt to
                # the same fixed exponent range on validation blocks.
                trials = [(nmse(validation, find_nearest(vectors >> shift, validation)[0]), shift)
                          for shift in range(6)]
                shift = min(trials)[1]
                name = f"{family} (calibrated)"
                baseline_validation.extend(dict(method=name, bits_per_block=bits, latent=None,
                                                seed_bits=seed_bits, coefficient_bits=None,
                                                output_shift=candidate, validation_nmse=value,
                                                selected=candidate == shift)
                                           for value, candidate in trials)
                adjusted = vectors >> shift
                errors, seeds, *_ = find_nearest(adjusted, holdout)
                record(name, bits, errors, vectors=adjusted, seeds=seeds,
                       note=f"Validation picked arithmetic output shift {shift} of 0–5; "
                       f"{costs[family]} plus output and scale")
        reconstructed, overhead = lloyd_max(calibration, holdout, bits)
        record("Lloyd–Max", bits, squared_error(holdout, reconstructed), overhead=overhead,
               note="FP16 levels per matrix; one lookup per weight")
        config = select_seedlm(bits)
        if config is not None:
            record("Refined SeedLM", bits, seeded_errors(config, holdout),
                   note=f"{config.seed_bits}-bit seed; {config.latent} bases with "
                   f"{config.coefficient_bits}-bit coefficients; "
                   f"{7 * config.latent} integer operations/weight plus basis normalization, "
                   "weighted sum and exponent scaling")
        for name, vectors in diagnostic_codebooks(seed_bits, calibration).items():
            errors, seeds, *_ = find_nearest(vectors, holdout)
            record(name, bits, errors, vectors=vectors, seeds=seeds, overhead=vectors.nbytes,
                   note="Explicit INT8 codebook; one eight-byte codeword read per block")
        if seed_bits == args.seed_bits:
            ids = np.arange(1 << seed_bits, dtype=np.uint32).astype(np.uint8).view(np.int8)
            for name, vectors in (
                ("Constant", np.zeros((1, WIDTH), dtype=np.int8)),
                ("Seed only", np.broadcast_to(ids[:, None], (len(ids), WIDTH))),
                ("Index only", np.arange(WIDTH, dtype=np.int8)[None, :]),
            ):
                errors, seeds, *_ = find_nearest(vectors, holdout)
                record(name, bits, errors, vectors=vectors, seeds=seeds, note="Degenerate control")
        print(f"Completed {bits / WIDTH:g} bpw holdout", flush=True)

    write_csv(out / "holdout.csv", final_rows)
    write_csv(out / "baseline_validation.csv", baseline_validation)
    write_csv(out / "by_projection.csv", projection_rows)
    (out / "diagnostics.json").write_text(json.dumps(details, separators=(",", ":")) + "\n")
    (out / "winners.json").write_text(json.dumps(programs, indent=2) + "\n")
    return chosen


def search_benchmark(args, samples, chosen, out):
    if not chosen:
        return
    program = chosen[max(chosen)][1]
    timings = []
    for bits in (6, 8, 10, 12, 14, 16):
        vectors = bitvector.codebook(program, bits, WIDTH)
        for count in (1, 42, 96):
            blocks = samples["screen"][:count]
            synthesized.exact_search(vectors, blocks)
            measures = []
            for _ in range(5):
                tick = time.perf_counter()
                synthesized.exact_search(vectors, blocks)
                measures.append(time.perf_counter() - tick)
            elapsed = float(np.median(measures))
            timings.append(dict(seed_bits=bits, seeds=len(vectors), encodings_per_block=len(vectors) * 16,
                                blocks=count, median_seconds=elapsed,
                                seed_block_pairs_per_second=len(vectors) * count / elapsed))
    write_csv(out / "search_scaling.csv", timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ops", nargs="+", choices=list(bitvector.OPERATIONS), default=list(bitvector.OPERATIONS))
    parser.add_argument("--constants", nargs="+", type=lambda value: int(value, 0), default=grammar.CONSTANTS)
    parser.add_argument("--seed-bits", type=int, default=12)
    parser.add_argument("--max-cost", type=int, default=49, help="Maximum scalar operations per eight-weight block")
    parser.add_argument("--three-limit", type=int, default=240_000, help="Maximum live 3-op DAGs, 0 for exhaustive")
    parser.add_argument("--finalists", type=int, default=64)
    args = parser.parse_args()
    args.ops = tuple(dict.fromkeys(args.ops))
    args.constants = tuple(dict.fromkeys(args.constants))
    args.max_instructions = min(3, (args.max_cost - grammar.SCALE_SETUP) // WIDTH - grammar.OUTPUT_COST)
    if (not 1 <= args.seed_bits <= 16 or args.max_instructions < 1 or args.three_limit < 0 or args.finalists < 1
            or not all(0 <= value <= bitvector.MASK for value in args.constants)):
        parser.error("Invalid seed width, instruction cost, enumeration limit, finalist count or uint32 literal")
    args.out.mkdir(parents=True, exist_ok=True)
    samples, manifest = sample_weights(args.checkpoint)
    weights = Weights(args.checkpoint)
    shapes = [weights.tensors[f"model.layers.{layer}.{group}.{projection}.weight"][2]["shape"]
              for layer in LAYERS for projection, group in PROJECTIONS.items()]
    parameter_count = sum(rows * cols for rows, cols in shapes)
    digests = {}
    for path in sorted({entry[0] for entry in weights.tensors.values()}):
        with path.open("rb") as handle:
            digests[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    (args.out / "config.json").write_text(json.dumps(dict(
        **vars(args), block_width=WIDTH, exponent_bits=4, sample_seed=SPLIT_SEED,
        enumeration_seed=ENUMERATION_SEED, layers=LAYERS, checkpoint_sha256=digests,
        split_blocks={name: len(blocks) for name, blocks in samples.items()},
        sample_matrix_weight_count=parameter_count,
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
    ), default=str, indent=2) + "\n")
    (args.out / "samples.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    finalists = run_enumeration(args, samples, args.out)
    winners = compare(args, samples, finalists, args.out, parameter_count)
    search_benchmark(args, samples, winners, args.out)


if __name__ == "__main__":
    main()
