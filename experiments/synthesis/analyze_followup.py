"""Reproduce geometry correlations and paired final-set bootstrap for Experiment 02."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.synthesis import miniature
from experiments.synthesis.followup import final_sample, geometry, scalar_program
from experiments.synthesis.guided import parse_program


def nearest_per_tile(book, targets):
    return np.min(np.sum((targets[:, None, :].astype(np.float64)
                          - book[None, :, :].astype(np.float64)) ** 2, axis=2), axis=1)


def main():
    previous = Path("experiments/results/miniature-codebook")
    out = Path("experiments/results/miniature-codebook-followup")
    summary = json.loads((out / "summary.json").read_text())
    guided = json.loads((out / "guided.json").read_text())
    original = json.loads((previous / "summary.json").read_text())
    manifest = json.loads((previous / "sample_manifest.json").read_text())
    final, _ = final_sample("data/smollm2-135m.safetensors", manifest)
    levels = np.asarray(original["baselines"]["lloyd_max_scalar"]["levels"])
    winner = guided["winners"][2]
    book = miniature.code(parse_program(winner["program"])).astype(np.float64) * winner["scale"]
    scalar = scalar_program(np.arange(256), levels)
    synthesized_errors = nearest_per_tile(book, final)
    scalar_errors = nearest_per_tile(scalar, final)
    energy = np.sum(final.astype(np.float64) ** 2)
    rng = np.random.default_rng(2602)
    bootstrap = []
    for _ in range(2000):
        chosen = rng.integers(0, len(final), len(final))
        bootstrap.append((synthesized_errors[chosen].sum() - scalar_errors[chosen].sum()) /
                         np.sum(final[chosen].astype(np.float64) ** 2))
    rows = list(csv.DictReader((out / "optimized_ranked.csv").open()))
    y = np.asarray([float(row["fitted_calibration_nmse"]) for row in rows])
    associations = {}
    for feature in ("coord_rms_cv", "coord_zero_fraction", "covariance_condition",
                    "avalanche_mean_l2", "avalanche_cv"):
        x = np.asarray([float(row[feature]) if row[feature] else np.nan for row in rows])
        valid = np.isfinite(x)
        associations[feature] = {
            "all": float(spearmanr(x[valid], y[valid]).statistic),
            "top_decile": float(spearmanr(x[:len(x)//10], y[:len(y)//10]).statistic)}
    programs = {
        "experiment_01_winner": original["winners"]["2"]["program"],
        "optimized_two_op": summary["winners"]["2"]["program"],
        "guided_three_op": guided["winners"][0]["program"],
        "guided_five_op": winner["program"]}
    analysis = {
        "geometry_score_spearman": associations,
        "winner_geometry": {key: geometry(miniature.code(parse_program(value)))
                            for key, value in programs.items()},
        "paired_final_difference": {
            "five_op_minus_scalar_nmse": float((synthesized_errors.sum() - scalar_errors.sum()) / energy),
            "tile_bootstrap_95_percent_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "bootstrap_replicates": 2000, "bootstrap_seed": 2602}}
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis["paired_final_difference"], indent=2))


if __name__ == "__main__":
    main()
