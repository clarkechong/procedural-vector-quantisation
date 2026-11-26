"""Streamed CPU reference audit for real 20- and 24-bit generator blocks."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from experiments.encoding.synthesized import exact_program_search
from experiments.generator_algebra.bitvector import Instruction
from experiments.manifests.weights import Weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--winners", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-batch", type=int, default=65536)
    parser.add_argument("--extended", action="store_true",
                        help="Audit every projection at layers 0, 15 and 29")
    args = parser.parse_args()
    selected = json.loads(args.winners.with_name("selection.json").read_text())["3"]
    winner = json.loads(args.winners.read_text())[selected]
    program = tuple(Instruction(row["op"], tuple(row["args"])) for row in winner["instructions"])
    checkpoint = Weights(args.checkpoint)
    samples = ((0, "q_proj", 0, 0), (15, "up_proj", 7, 3), (29, "down_proj", 17, 11))
    if args.extended:
        from experiments.manifests.weights import PROJECTIONS
        samples = tuple((layer, name, 7 + layer // 5, 3 + list(PROJECTIONS).index(name))
                        for layer in (0, 15, 29) for name in PROJECTIONS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("layer", "projection", "row", "block", "seed_bits", "seed", "exponent",
                         "sse", "decoded", "seconds"))
        for layer, projection, row, block in samples:
            target = checkpoint.get(layer, projection)[row, block * 8:(block + 1) * 8]
            prior = None
            for bits in (20, 24):
                tick = time.perf_counter()
                errors, seeds, exponents, decoded = exact_program_search(
                    program, target[None], bits, args.seed_batch)
                elapsed = time.perf_counter() - tick
                if prior is not None and errors[0] > prior + 1e-12:
                    raise AssertionError("24-bit superset increased reconstruction SSE")
                prior = errors[0]
                writer.writerow((layer, projection, row, block, bits, int(seeds[0]),
                                 int(exponents[0]), float(errors[0]),
                                 json.dumps(decoded[0].tolist()), elapsed))
                handle.flush()
                print(f"{layer} {projection} row {row} block {block}: {bits} bits, "
                      f"seed {seeds[0]}, exponent {exponents[0]}, SSE {errors[0]:.8g}, "
                      f"{elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
