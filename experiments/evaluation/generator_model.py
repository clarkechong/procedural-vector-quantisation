"""Full-model WikiText-2 evaluation of a frozen generator and paper SeedLM."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import scipy

from experiments.encoding import synthesized
from experiments.encoding.indexed_generator import DirectEncoder
from experiments.encoding.paper_seedlm import PaperEncoder
from experiments.encoding.seedlm import PUBLISHED, decode, pack, search
from experiments.evaluation.seedlm_model import add_evaluation_arguments, digest, run
from experiments.generator_algebra import bitvector
from experiments.manifests.weights import PROJECTIONS, Weights


RESULTS = Path(__file__).resolve().parents[1] / "results"
DEFAULT_WINNERS = RESULTS / "generator-search-smollm2" / "winners.json"

METHODS = ("original", "synthesized_2", "synthesized_2p5", "seedlm_paper_3", "seedlm_paper_4",
           "seedlm_all_seeds_3", "seedlm_all_seeds_4")


def audit(args, program):
    checkpoint = Weights(args.checkpoint)
    rng = np.random.default_rng(9221)
    spans = []
    for layer in (0, 15, 29):
        for name in PROJECTIONS:
            matrix = checkpoint.get(layer, name)
            rows = rng.choice(matrix.shape[0], 4, replace=False)
            for row in rows:
                start = 24 * rng.integers(matrix.shape[1] // 24 - 7)
                spans.append(matrix[row, start:start + 192])
    spans = np.stack(spans)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "encoder_audit.csv").open("w", newline="") as output:
        w = csv.writer(output)
        w.writerow(("bpw", "blocks", "pool", "indexed_nmse", "exhaustive_nmse", "same_error_fraction",
                    "max_error_decrease", "index_seconds", "indexed_seconds", "exhaustive_seconds",
                    "bounded_seconds", "bounded_exact_fields"))
        for config in PUBLISHED:
            tick = time.perf_counter()
            encoder = PaperEncoder(config, args.pool, args.library, args.threads, projection_bound=True)
            setup_seconds = time.perf_counter() - tick
            width = config.block
            blocks = spans.reshape(-1, width)
            tick = time.perf_counter()
            indexed = encoder.fields(blocks, use_bound=False)
            indexed_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            bounded = encoder.fields(blocks)
            bounded_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            exact = encoder.fields(blocks, exhaustive=True)
            exact_seconds = time.perf_counter() - tick
            def errors(fields):
                decoded = decode(config, pack(config, *fields), len(blocks))
                return np.square(blocks - decoded, dtype=np.float64).sum(axis=1)
            e_indexed, e_exact = errors(indexed), errors(exact)
            energy = np.square(blocks, dtype=np.float64).sum()
            _, _, reference = next(search(config, blocks[:12], rules=("paper",)))
            np.testing.assert_allclose(e_exact[:12], reference[0], rtol=5e-5, atol=2e-8)
            assert (e_exact <= e_indexed + 2e-8).all()
            same_fields = all(np.array_equal(a, b) for a, b in zip(exact, bounded))
            if not same_fields:
                raise RuntimeError("Projection pruning did not reproduce the exhaustive encoder")
            row = (config.rate, len(blocks), args.pool, e_indexed.sum() / energy, e_exact.sum() / energy,
                   np.isclose(e_indexed, e_exact, rtol=3e-6, atol=1e-10).mean(),
                   np.max(e_indexed - e_exact), setup_seconds, indexed_seconds, exact_seconds, bounded_seconds, same_fields)
            print(row, flush=True)
            w.writerow(row)
            output.flush()


    matrix = checkpoint.get(15, "up_proj").reshape(-1, 8)
    selected = np.random.default_rng(28529).choice(len(matrix), 192, replace=False)
    blocks = matrix[selected]
    with (args.out / "direct_exactness.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("seed_bits", "blocks", "max_sse_difference", "same_vectors"))
        for bits in (12, 16):
            encoder = DirectEncoder(program, bits, args.threads)
            payload = encoder.encode(blocks)
            indexed = encoder.decode(payload, len(blocks))
            scalar = synthesized.decode(program, payload, bits, 8, count=len(blocks))
            np.testing.assert_array_equal(indexed, scalar)
            errors, _, _, exhaustive = synthesized.exact_search(encoder.unscaled, blocks)
            observed = np.square(blocks.astype(np.float64) - indexed).sum(axis=1)
            np.testing.assert_array_equal(observed, errors)
            writer.writerow((bits, len(blocks), np.max(np.abs(errors - observed)),
                             np.array_equal(indexed, exhaustive)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--winners", type=Path, default=DEFAULT_WINNERS)
    parser.add_argument("--library", type=Path, help="Native encoder for SeedLM methods or --audit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS[:3] + METHODS[-2:])
    parser.add_argument("--pool", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--calibration-tokens", type=int, default=512)
    parser.add_argument("--audit", action="store_true")
    add_evaluation_arguments(parser)
    args = parser.parse_args()
    if not (1 <= args.pool <= 65535 and args.threads > 0 and args.context >= 2 and args.batch > 0
            and args.calibration_tokens >= 0 and args.tokens >= 0 and args.tokens != 1 and args.windows >= 0):
        parser.error("Invalid pool, thread, batch, context, or token count")
    if (args.audit or any(method.startswith("seedlm_") for method in args.methods)) and args.library is None:
        parser.error("Pass --library for SeedLM methods or --audit")
    winner_id = json.loads(args.winners.with_name("selection.json").read_text())["3"]
    winner = json.loads(args.winners.read_text())[winner_id]
    program = tuple(bitvector.Instruction(row["op"], tuple(row["args"])) for row in winner["instructions"])
    if args.audit:
        audit(args, program)
        return
    if args.data is None or "original" in args.methods[1:]:
        parser.error("Supply the evaluation data and place original first if requested")
    factories = {
        "synthesized_2": lambda: DirectEncoder(program, 12, args.threads),
        "synthesized_2p5": lambda: DirectEncoder(program, 16, args.threads),
        "seedlm_paper_3": lambda: PaperEncoder(PUBLISHED[0], args.pool, args.library, args.threads),
        "seedlm_paper_4": lambda: PaperEncoder(PUBLISHED[1], args.pool, args.library, args.threads),
        "seedlm_all_seeds_3": lambda: PaperEncoder(PUBLISHED[0], args.pool, args.library, args.threads,
                                                   projection_bound=True),
        "seedlm_all_seeds_4": lambda: PaperEncoder(PUBLISHED[1], args.pool, args.library, args.threads,
                                                   projection_bound=True),
    }
    args.seeds, args.shortlist, args.neighborhood = 255, 8, 192  # Legacy harness defaults.
    run(args, factories, dict(generator=winner_id, generator_program=bitvector.expression(program),
                             native_library_sha256=digest(args.library) if args.library else None,
                             paper_indexed_seeds=args.pool, paper_exhaustive_seeds=65535,
                             seedlm_block_layout="contiguous row-major matrix with zero-padded tail",
                             scipy=scipy.__version__))


if __name__ == "__main__":
    main()
