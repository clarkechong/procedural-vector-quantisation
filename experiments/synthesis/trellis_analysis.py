"""Matched-subset scalar controls for Experiment 03 rate–distortion data."""

import json
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

from experiments.encoding.codebooks import hadamard_rows, one_mad
from experiments.generator_algebra import bitvector
from experiments.synthesis.guided import parse_program
from experiments.synthesis.simplify import fold_multipliers
from experiments.synthesis.trellis import encode, linear_map
from experiments.synthesis.trellis_experiment import (LENGTHS, OUT, fit_scale,
                                                      sample_rows, score)


def scalar_control(calibration, test, levels):
    flat = calibration.ravel()
    # Deterministic quantile initialization followed by Lloyd updates.
    start = np.quantile(flat, (np.arange(levels) + 0.5) / levels)
    centers, _ = kmeans2(flat, start, iter=50, minit="matrix")
    reconstructed = centers[np.abs(test[..., None] - centers).argmin(-1)]
    return dict(levels=np.sort(centers).tolist(),
                test_nmse=float(np.square(test - reconstructed).sum() /
                                np.square(test).sum()), test_rows=len(test))


def main():
    prior = Path("experiments/results/miniature-codebook")
    followup = Path("experiments/results/miniature-codebook-followup")
    old = json.loads((prior / "sample_manifest.json").read_text())
    final = json.loads((followup / "final_manifest.json").read_text())
    samples, manifest = sample_rows("data/smollm2-135m.safetensors", (old, final))
    assert manifest == json.loads((OUT / "sample_manifest.json").read_text())
    results = json.loads((OUT / "results.json").read_text())
    existing = {(row["length"], row["method"]) for row in results["rate_distortion"]}
    for length in LENGTHS:
        if (length, "byte_120_L4_k2") not in existing:
            calibration = samples[length]["calibration"]
            test = samples[length]["test"]
            mapping = linear_map(4, 120)
            scale = fit_scale(calibration, 4, 2, mapping)
            trained = score(calibration, 4, 2, mapping * scale)
            measured = score(test, 4, 2, mapping * scale)
            results["rate_distortion"].append(dict(length=length, method="byte_120_L4_k2",
                state_bits=4, stride=2, payload_bits=4 + 2 * (length - 1),
                payload_bpw=(4 + 2 * (length - 1)) / length, scale=scale,
                calibration_nmse=trained["nmse"], test_nmse=measured["nmse"],
                test_rows=len(test), test_weights=int(test.size)))
    for length in (64, 128, 256, 1024):
        if (length, "one_mad_L16_k2_rotated") not in existing:
            calibration = hadamard_rows(samples[length]["calibration"][:4])
            test = hadamard_rows(samples[length]["test"][:4])
            mapping = one_mad(16)
            scale = fit_scale(calibration, 16, 2, mapping)
            trained = score(calibration, 16, 2, mapping * scale)
            measured = score(test, 16, 2, mapping * scale)
            results["rate_distortion"].append(dict(length=length,
                method="one_mad_L16_k2_rotated", state_bits=16, stride=2,
                payload_bits=16 + 2 * (length - 1),
                payload_bpw=(16 + 2 * (length - 1)) / length, scale=scale,
                calibration_nmse=trained["nmse"], test_nmse=measured["nmse"],
                test_rows=len(test), test_weights=int(test.size),
                transform="fixed signed 64-point Hadamard groups; orthogonal inverse"))
    results["rate_distortion"].sort(key=lambda row: (row["length"], row["method"]))
    guided = json.loads((followup / "guided.json").read_text())
    simplifications = []
    for winner in guided["winners"]:
        program = parse_program(winner["program"])
        reduced, folds = fold_multipliers(program)
        original_book = bitvector.codebook(program, 8, 4)
        reduced_book = bitvector.codebook(reduced, 8, 4)
        assert np.array_equal(original_book, reduced_book)
        simplifications.append(dict(original=bitvector.expression(program),
                                    simplified=bitvector.expression(reduced),
                                    original_length=len(program), effective_length=len(reduced),
                                    constant_folds=folds, exact_equivalence=True))
    results["guided_winner_simplifications"] = simplifications
    controls = []
    for length in LENGTHS:
        for rows in (4, 12, len(samples[length]["test"])):
            if rows > len(samples[length]["test"]):
                continue
            calibration = samples[length]["calibration"][:rows]
            test = samples[length]["test"][:rows]
            mapping = linear_map(5, 120)
            for stride in (1, 2):
                scale = fit_scale(calibration, 5, stride, mapping)
                window = score(test, 5, stride, mapping * scale)
                controls.append(dict(length=length, rows=rows,
                                     method=f"discovered_120_k{stride}",
                                     bpw=(5 + stride * (length - 1)) / length,
                                     test_nmse=window["nmse"]))
            for levels in (2, 4):
                scalar = scalar_control(calibration, test, levels)
                controls.append(dict(length=length, rows=rows,
                                     method=f"scalar_{levels}_levels", bpw=np.log2(levels),
                                     test_nmse=scalar["test_nmse"], levels=scalar["levels"]))
    results["matched_scalar_controls"] = controls
    four_cal = samples[4]["calibration"]
    four_test = samples[4]["test"]
    four_mapping = linear_map(5, 120)
    four_scale = fit_scale(four_cal, 5, 1, four_mapping)
    window_errors = np.asarray([encode(row, 5, 1, four_mapping * four_scale)[2]
                                for row in four_test])
    scalar_levels = np.asarray(next(row["levels"] for row in controls
                                    if row["length"] == 4 and row["rows"] == len(four_test)
                                    and row["method"] == "scalar_4_levels"))
    scalar_values = scalar_levels[np.abs(four_test[..., None] - scalar_levels).argmin(-1)]
    scalar_errors = np.square(four_test - scalar_values).sum(axis=1)
    energies = np.square(four_test).sum(axis=1)
    rng = np.random.default_rng(303)
    indices = rng.integers(0, len(four_test), size=(2000, len(four_test)))
    differences = ((scalar_errors[indices].sum(axis=1) -
                    window_errors[indices].sum(axis=1)) /
                   energies[indices].sum(axis=1))
    results["paired_four_weight"] = dict(
        scalar_minus_discovered_nmse=float((scalar_errors.sum() - window_errors.sum()) /
                                           energies.sum()),
        row_bootstrap_95_percent_interval=np.quantile(differences, (0.025, 0.975)).tolist(),
        bootstrap_replicates=2000, rows=len(four_test))
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps([row for row in controls if row["rows"] == 4], indent=2))


if __name__ == "__main__":
    main()
