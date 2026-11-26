"""Summarize the frozen generator search, including paired holdout comparisons."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def paired_comparisons(out, primary, winner, holdout):
    projections = read_csv(out / "by_projection.csv")
    methods = {
        row["method"]: np.asarray([
            (float(entry["sse"]), float(entry["energy"])) for entry in projections
            if entry["method"] == row["method"] and float(entry["bits_per_weight"]) == primary
        ])
        for row in holdout
    }
    candidate = methods[winner]
    selection = np.random.default_rng(5241).integers(len(candidate), size=(10_000, len(candidate)))
    comparisons = []
    for name, baseline in methods.items():
        if name == winner:
            continue
        np.testing.assert_array_equal(candidate[:, 1], baseline[:, 1])
        delta = candidate[:, 0] - baseline[:, 0]
        denominators = candidate[:, 1]
        draws = delta[selection].sum(axis=1) / denominators[selection].sum(axis=1)
        lower, upper = np.quantile(draws, (.025, .975))
        comparisons.append(dict(candidate=winner, comparator=name, paired_nmse_difference=delta.sum() /
                                denominators.sum(), bootstrap_lower=lower, bootstrap_upper=upper))
    if comparisons:
        with (out / "paired_comparisons.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, comparisons[0])
            writer.writeheader()
            writer.writerows(comparisons)
    return comparisons


def plot(out, all_rows, primary):
    preferred = ("Synthesized 1-op", "Synthesized 2-op", "Synthesized 3-op",
                 "LFSR", "Xorshift", "SplitMix", "Lloyd–Max", "Refined SeedLM",
                 "Gaussian codebook", "Trained VQ")
    fig, (rate_axis, cost_axis) = plt.subplots(1, 2, figsize=(12, 5),
                                               gridspec_kw={"width_ratios": [1.65, 1]})
    fig.set_facecolor("#f9faf8")
    for axis in (rate_axis, cost_axis):
        axis.set_facecolor("#f9faf8")
        axis.grid(alpha=.2)
    colors = ("#8dbab0", "#438c80", "#05493e", "#708090", "#a78262", "#68467d", "#b64526",
              "#214495", "#bf905e", "#c75174")
    for method, color in zip(preferred, colors):
        rows = [row for row in all_rows if row["method"] == method]
        if not rows:
            continue
        generated = method.startswith("Synthesized")
        rate_axis.plot([float(row["bits_per_weight"]) for row in rows],
                       [float(row["nmse"]) for row in rows],
                       marker="o" if generated else ".", color=color,
                       linewidth=2 if generated else 1.3,
                       linestyle="-" if generated else "--", label=method)
        if generated:
            for row in rows:
                if float(row["bits_per_weight"]) != primary:
                    continue
                x, y = int(row["cost_per_block"]), float(row["nmse"])
                cost_axis.scatter(x, y, color=color, s=75, zorder=3)
                offset = (-10, 8) if x >= 49 else (5, 8)
                cost_axis.annotate(method.replace("Synthesized ", ""), (x, y), xytext=offset,
                                   textcoords="offset points", color=color)
    for name, marker, color in (("Xorshift (calibrated)", "x", "#a78262"),
                                ("SplitMix (calibrated)", "+", "#68467d")):
        rows = [row for row in all_rows if row["method"] == name]
        if rows:
            rate_axis.scatter([float(row["bits_per_weight"]) for row in rows],
                              [float(row["nmse"]) for row in rows],
                              marker=marker, color=color, label=name, zorder=4, s=42)
    rate_axis.axvline(primary, color="#3f3f3f", alpha=.24)
    rate_axis.set(xlabel="Payload bits per weight", ylabel="Pooled holdout NMSE",
                  title="Rate–distortion on fresh SmolLM2 weights", yscale="log")
    rate_axis.set_xticks(sorted({float(row["bits_per_weight"]) for row in all_rows}))
    rate_axis.set_yticks((0.1, 0.2, 0.3, 0.5, 0.7))
    rate_axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    rate_axis.legend(fontsize=8, ncols=2)
    cost_axis.set(xlabel="Scalar operations per block (estimated)", ylabel="Pooled holdout NMSE",
                  title=f"Synthesized generators at {primary:g} bpw", xlim=(29, 53), ylim=(.16, .61), xticks=(33, 41, 49))
    fig.tight_layout(pad=2)
    fig.savefig(out / "rate-distortion.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    out = args.directory
    config = json.loads((out / "config.json").read_text())
    primary = (config["seed_bits"] + config["exponent_bits"]) / config["block_width"]
    rows = read_csv(out / "holdout.csv")
    matching = [row for row in rows if float(row["bits_per_weight"]) == primary]
    selected = json.loads((out / "selection.json").read_text())
    report = []
    if selected:
        validation = {row["program_id"]: row for row in read_csv(out / "validation.csv")}
        # The cross-cost winner is chosen by validation alone.
        winner = min(selected.values(), key=lambda name: (float(validation[name]["validation_nmse"]), name))
        report = paired_comparisons(out, primary, f"Synthesized {winner.split('-')[0]}-op", matching)
    plot(out, rows, primary)
    for row in matching:
        print(row["method"], row["nmse"], row["unique_unscaled"], row["unique_scaled"], row["note"])
    for row in report:
        print(row)


if __name__ == "__main__":
    main()
