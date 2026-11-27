"""Bounded beam search for three- through six-instruction signed-byte programs."""

import argparse
import csv
import json
from pathlib import Path
import re
import time

import numpy as np

from experiments.generator_algebra import bitvector
from experiments.synthesis import grammar, miniature
from experiments.synthesis.followup import EXTRA_CONSTANTS, final_sample, optimize_scale


def parse_program(expression):
    steps = []
    for item in expression.split("; "):
        match = re.fullmatch(r"r\d+=(\w+)\((.*)\)", item)
        if not match:
            raise ValueError(expression)
        args = tuple(int(arg, 16) if arg.startswith("0x") else arg
                     for arg in match.group(2).split(", "))
        steps.append(bitvector.Instruction(match.group(1), args))
    return tuple(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/smollm2-135m.safetensors")
    parser.add_argument("--previous", default="experiments/results/miniature-codebook")
    parser.add_argument("--followup", default="experiments/results/miniature-codebook-followup")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam", type=int, default=24)
    args = parser.parse_args()
    start = time.monotonic()
    previous = Path(args.previous)
    out = Path(args.followup)
    ranked = list(csv.DictReader((out / "optimized_ranked.csv").open()))
    initial = [(parse_program(row["program"]), float(row["fitted_calibration_nmse"]))
               for row in ranked if int(row["instructions"]) == 2][:args.beam]
    samples, _ = miniature.sample(args.checkpoint)
    cal, hold = samples["calibration"], samples["heldout"]
    final, _ = final_sample(args.checkpoint, json.loads((previous / "sample_manifest.json").read_text()))
    rng = np.random.default_rng(20261127)
    screen = cal[rng.choice(len(cal), 1024, replace=False)]
    scale = json.loads((previous / "summary.json").read_text())["global_scale"]
    # A separate two-step beam audit measures the cost of pruning prefixes.
    one_step = [(parse_program(row["program"]), float(row["fitted_calibration_nmse"]))
                for row in ranked if int(row["instructions"]) == 1]
    audit_prefixes = [program for program, _ in one_step[:args.beam]]
    audit_choices = [instruction for instruction in grammar.choices(
        1, miniature.OPS, miniature.CONSTANTS) if "r0" in instruction.args]
    audit_seen = set()
    audit_rows = []
    for prefix in audit_prefixes:
        for instruction in audit_choices:
            program = (*prefix, instruction)
            book = miniature.code(program)
            key = book.tobytes()
            if key in audit_seen:
                continue
            audit_seen.add(key)
            audit_rows.append((program, book))
    audit_scored = []
    for index in range(0, len(audit_rows), 16):
        batch = audit_rows[index:index+16]
        scores, scales = optimize_scale([book for _, book in batch], cal,
                                        args.device, scale)
        audit_scored.extend((float(score), float(fitted), program)
                            for (program, _), score, fitted in zip(batch, scores, scales))
    audit_scored.sort(key=lambda row: row[0])
    exact_best = next(row for row in ranked if int(row["instructions"]) == 2)
    beam_audit = {"one_instruction_distinct": len(one_step),
                  "kept_one_instruction_prefixes": len(audit_prefixes),
                  "two_instruction_syntactic_trials": len(audit_prefixes)*len(audit_choices),
                  "two_instruction_distinct_scored": len(audit_rows),
                  "guided_best": {"program": bitvector.expression(audit_scored[0][2]),
                                  "calibration_nmse": audit_scored[0][0]},
                  "exhaustive_best": {"program": exact_best["program"],
                                      "calibration_nmse": float(exact_best["fitted_calibration_nmse"])},
                  "gap_nmse": audit_scored[0][0]-float(exact_best["fitted_calibration_nmse"]),
                  "note": "All distinct one-op prefixes were scored; only their best 24 were expanded. The exhaustive comparator uses the same fitted-scale procedure."}
    (out / "beam_audit.json").write_text(json.dumps(beam_audit, indent=2) + "\n")
    beam = [p for p, _ in initial]
    # The bit-field path is included as a seeded positive-control branch;
    # beam discovery is not claimed for this hand-inserted prefix.
    scalar_prefix = (bitvector.Instruction("SHL", ("i", 1)),
                     bitvector.Instruction("SHR", ("s", "r0")))
    if scalar_prefix not in beam:
        beam.append(scalar_prefix)
    seen = set()
    coverage, finalists = [], []
    for length in range(3, 7):
        choices = [instruction for instruction in grammar.choices(
            length - 1, miniature.OPS, miniature.CONSTANTS + EXTRA_CONSTANTS)
            if f"r{length-2}" in instruction.args]
        rows = []
        books = []
        for prefix in beam:
            for instruction in choices:
                program = (*prefix, instruction)
                book = miniature.code(program)
                key = book.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"program": program, "length": length})
                books.append(book)
        for index in range(0, len(rows), 24):
            scores, scales = optimize_scale(books[index:index+24], screen,
                                            args.device, scale, iterations=3)
            for row, score, fitted in zip(rows[index:index+24], scores, scales):
                row["screen_nmse"] = float(score)
                row["screen_scale"] = float(fitted)
        rows.sort(key=lambda row: row["screen_nmse"])
        # Rerank 96 candidates on all calibration tiles to avoid selection by
        # the smaller screen. Final tiles are consulted only after selection.
        chosen = rows[:96]
        for index in range(0, len(chosen), 16):
            subset = chosen[index:index+16]
            scores, scales = optimize_scale([miniature.code(row["program"]) for row in subset],
                                            cal, args.device, scale)
            for row, score, fitted in zip(subset, scores, scales):
                row["calibration_nmse"] = float(score)
                row["scale"] = float(fitted)
        chosen.sort(key=lambda row: row["calibration_nmse"])
        winner = chosen[0]
        book = miniature.code(winner["program"])
        winner["heldout_nmse"] = miniature.score_one(book, hold, winner["scale"], args.device)
        winner["final_nmse"] = miniature.score_one(book, final, winner["scale"], args.device)
        finalists.append({key: bitvector.expression(value) if key == "program" else value
                          for key, value in winner.items()})
        coverage.append({"instructions": length, "incoming_beam": len(beam),
                         "choices_per_prefix": len(choices), "syntactic_trials": len(beam)*len(choices),
                         "new_distinct_codebooks": len(rows), "full_calibration_reranked": len(chosen)})
        beam = [row["program"] for row in chosen[:args.beam]]
        print(f"length {length}: {winner['calibration_nmse']:.6f} calibration, "
              f"{winner['heldout_nmse']:.6f} heldout; {len(rows)} screened", flush=True)
    result = {"method": "24-wide beam from optimized two-op programs plus hand-inserted scalar-extraction prefix; 1024-tile screen, top-96 full calibration rerank",
              "coverage": coverage, "winners": finalists,
              "elapsed_seconds": time.monotonic()-start,
              "limits": "Nonexhaustive; beam and screening can miss better programs; fixed signed-byte output cannot itself perform FP16 LUT4."}
    (out / "guided.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
