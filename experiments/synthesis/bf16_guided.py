"""Bounded direct-BF16 search with BF16 masks and bit-pattern constants."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.generator_algebra import bitvector
from experiments.synthesis import grammar, miniature
from experiments.synthesis.followup import EXTRA_CONSTANTS, final_sample


OPS = ("ADD", "SUB", "MUL", "AND", "OR", "XOR", "SHL", "SHR")
CONSTANTS = miniature.CONSTANTS + EXTRA_CONSTANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/smollm2-135m.safetensors")
    parser.add_argument("--previous", default="experiments/results/miniature-codebook")
    parser.add_argument("--followup", default="experiments/results/miniature-codebook-followup")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    start = time.monotonic()
    out = Path(args.followup)
    samples, _ = miniature.sample(args.checkpoint)
    cal, hold = samples["calibration"], samples["heldout"]
    final, _ = final_sample(args.checkpoint, json.loads((Path(args.previous) / "sample_manifest.json").read_text()))
    rng = np.random.default_rng(20261127)
    screen = cal[rng.choice(len(cal), 512, replace=False)]
    choices = tuple(grammar.choices(0, OPS, CONSTANTS))
    second = tuple(x for x in grammar.choices(1, OPS, CONSTANTS) if "r0" in x.args)
    seen = set()
    rows = []
    first = []
    for instruction in choices:
        book = miniature.bf16_code((instruction,))
        if book is None:
            continue
        key = book.tobytes()
        if key in seen:
            continue
        seen.add(key)
        first.append((instruction, book))
    screen_energy = float(np.square(screen.astype(np.float64)).sum())
    for index in range(0, len(first), 16):
        batch = first[index:index+16]
        scores = miniature.nearest_errors(np.stack([book for _, book in batch]),
                                          screen, 1, args.device, dtype=torch.float64)
        rows.extend((float(score / screen_energy), (instruction,))
                    for (instruction, _), score in zip(batch, scores))
    rows.sort(key=lambda item: item[0])
    # Retain top one-step programs and specific sign/exponent-mask prefixes.
    selected = [p for _, p in rows[:24]]
    for instruction, _ in first:
        if any(isinstance(arg, int) and arg in (0x8000, 0x7f80, 0x3c00, 0x3d00)
               for arg in instruction.args):
            program = (instruction,)
            if program not in selected:
                selected.append(program)
            if len(selected) == 48:
                break
    candidates = []
    for prefix in selected:
        for instruction in second:
            program = (*prefix, instruction)
            book = miniature.bf16_code(program)
            if book is None:
                continue
            key = book.tobytes()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((program, book))
    scored = []
    for index in range(0, len(candidates), 16):
        batch = candidates[index:index+16]
        scores = miniature.nearest_errors(np.stack([book for _, book in batch]),
                                          screen, 1, args.device, dtype=torch.float64)
        scored.extend((float(score / screen_energy), program, book)
                      for (program, book), score in zip(batch, scores))
    scored.sort(key=lambda item: item[0])
    finalists = []
    for _, program, book in scored[:96]:
        finalists.append((miniature.score_one(book, cal, 1, args.device), program, book))
    finalists.sort(key=lambda item: item[0])
    winner = finalists[0]
    hold_nmse = float(miniature.nearest_errors(winner[2][None], hold, 1,
                                               args.device, torch.float64)[0] /
                      np.square(hold.astype(np.float64)).sum())
    final_nmse = float(miniature.nearest_errors(winner[2][None], final, 1,
                                                args.device, torch.float64)[0] /
                       np.square(final.astype(np.float64)).sum())
    result = {"grammar": {"operations": OPS, "constants": CONSTANTS,
                          "description": "Low 16 output bits read as BF16; no learned scale"},
              "coverage": {"one_instruction_syntax": len(choices),
                           "one_instruction_valid_unique": len(first),
                           "selected_prefixes": len(selected),
                           "second_instruction_choices_per_prefix": len(second),
                           "two_instruction_syntactic_trials": len(selected)*len(second),
                           "two_instruction_valid_unique": len(candidates),
                           "full_calibration_reranked": len(finalists)},
              "winner": {"program": bitvector.expression(winner[1]),
                         "calibration_nmse": winner[0],
                         "heldout_nmse": hold_nmse,
                         "final_nmse": final_nmse},
              "elapsed_seconds": time.monotonic()-start,
              "limits": "Two-step search is guided by selected prefixes and screened on 512 calibration tiles; no exhaustiveness claim."}
    (out / "bf16_guided.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
