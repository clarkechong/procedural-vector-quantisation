"""Build rate-budget envelopes and paired summaries from a completed run."""

import csv
import gzip
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.rate_distortion.pilot import RATES


METHODS = {
    "procedural": "Mixed dictionary",
    "coordinate_only": "Coordinate atoms",
    "random_atoms": "Random atoms",
    "seeded_basis": "Seeded bases",
    "sparse": "Top-k sparsity",
    "uniform": "Uniform quantization",
    "nonuniform": "Lloyd–Max quantization",
    "svd": "Low rank",
}
DATASETS = {
    "original": "Original",
    "global_shuffle": "Global shuffle",
    "within_tile_shuffle": "Within-tile shuffle",
    "gaussian": "IID Gaussian",
    "injected": "Injected",
    "planted": "Planted",
    "original_order": "Original order",
    "random_order": "Random order",
    "fitted_order": "Fitted order",
}


def budget_envelopes(path):
    envelopes = defaultdict(lambda: np.ones(len(RATES)))
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["layer"]), row["projection"], row["granularity"],
                   row["dataset"], int(row["trial"]), row["method"])
            rate, error = float(row["bits_per_weight"]), float(row["nmse"])
            values = envelopes[key]
            if not float(row["energy"]):
                values.fill(0)
            start = np.searchsorted(RATES, rate)
            np.minimum(values[start:], error, out=values[start:])
    return envelopes


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def mean_curves(envelopes):
    grouped = defaultdict(list)
    for (_, _, granularity, dataset, _, method), curve in envelopes.items():
        grouped[granularity, dataset, method].append(curve)
    return {key: np.mean(values, axis=0) for key, values in grouped.items()}


def paired_rows(envelopes):
    null_pairs = (
        ("original", "global_shuffle"),
        ("original", "within_tile_shuffle"),
        ("original_order", "random_order"),
        ("fitted_order", "random_order"),
        ("fitted_order", "original_order"),
    )
    groups = defaultdict(dict)
    for (layer, projection, granularity, dataset, trial, method), values in envelopes.items():
        groups[granularity, method][layer, projection, dataset, trial] = values
    for (granularity, method), entries in sorted(groups.items()):
        for observed, null in null_pairs:
            trials = sorted({trial for _, _, dataset, trial in entries if dataset == null})
            paired = [(layer, projection, curve)
                      for (layer, projection, dataset, _), curve in entries.items()
                      if dataset == observed and trials
                      and all((layer, projection, null, trial) in entries for trial in trials)]
            if not paired:
                continue
            original = np.stack([curve for _, _, curve in paired])
            shuffles = np.stack([
                [entries[layer, projection, null, trial] for layer, projection, _ in paired]
                for trial in trials
            ])
            mean_original = original.mean(axis=0)
            null_means = shuffles.mean(axis=1)
            reference_mean = null_means.mean(axis=0)
            delta = reference_mean - mean_original
            wins = (shuffles.mean(axis=0) > original).sum(axis=0)
            p_values = (1 + (null_means <= mean_original).sum(axis=0)) / (len(trials) + 1)
            reference_sd = null_means.std(axis=0, ddof=1) if len(trials) > 1 else None
            for index, rate in enumerate(RATES):
                # A fitted ordering was learned, so random order is a
                # descriptive reference rather than an exchangeable null.
                p_value = p_values[index] if observed == "original" else ""
                yield (granularity, method, observed, null, rate, len(paired), len(trials),
                       mean_original[index], reference_mean[index], delta[index],
                       wins[index], reference_sd[index] if reference_sd is not None else "", p_value)


def plot_curves(out, means):
    granularities = list(dict.fromkeys(granularity for granularity, _, _ in means))
    focus = "16x16" if "16x16" in granularities else granularities[0]
    permutation_focus = next((g for g in [focus, *granularities]
                              if (g, "fitted_order", "procedural") in means), None)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")

    def plot(ax, entries, title, ylabel="Mean NMSE", zoom=False):
        entries = [(label, values) for label, values in entries if values is not None]
        for label, values in entries:
            ax.plot(RATES, values, marker=".", label=label)
        ax.set(xscale="log", xlabel="Rate budget (bits/weight)", ylabel=ylabel, title=title,
               ylim=(-0.025, 1.025), xticks=RATES, xticklabels=[str(x) for x in RATES])
        if zoom and entries:
            lo = min(values.min() for _, values in entries)
            hi = max(values.max() for _, values in entries)
            margin = max((hi - lo) * 0.06, 0.002)
            ax.set_ylim(lo - margin, hi + margin)
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(fontsize=8)

    plot(axes[0, 0], ((label, means.get((focus, "original", method)))
                     for method, label in METHODS.items()), f"Compression baselines · {focus}")
    controls = ("original", "global_shuffle", "within_tile_shuffle", "gaussian", "injected", "planted")
    plot(axes[0, 1], ((DATASETS[dataset], means.get((focus, dataset, "procedural")))
                     for dataset in controls), f"Mixed dictionary controls · {focus}")
    plot(axes[1, 0], ((g, means.get((g, "original", "procedural"))) for g in granularities),
         "Mixed dictionary by granularity")
    if permutation_focus:
        original = means[permutation_focus, "original_order", "procedural"]
        axes[1, 1].axhline(0, color="0.55", linewidth=1, linestyle="--")
        plot(axes[1, 1],
             ((DATASETS[dataset], means[permutation_focus, dataset, "procedural"] - original)
              for dataset in ("original_order", "fitted_order", "random_order")),
             f"Shared FFN ordering · {permutation_focus}",
             ylabel="Mean NMSE change from original order", zoom=True)
    else:
        plot(axes[1, 1], ((g, means.get((g, "planted", "procedural"))) for g in granularities),
             "Planted control by granularity")
    fig.savefig(out / "rate_distortion.png", dpi=160)
    plt.close(fig)


def write_report(out):
    envelopes = budget_envelopes(out / "curves.csv.gz")
    means = mean_curves(envelopes)
    if not means:
        raise ValueError("No compression measurements were produced")
    write_csv(out / "budget_curves.csv", ("granularity", "dataset", "method", "budget", "mean_nmse"),
              ((*key, rate, value) for key, values in sorted(means.items())
               for rate, value in zip(RATES, values)))
    write_csv(out / "paired_controls.csv",
              ("granularity", "method", "observed", "reference", "budget", "matrices", "trials",
               "observed_nmse", "reference_nmse", "reference_minus_observed", "matrix_wins",
               "reference_draw_sd", "mc_p"),
              paired_rows(envelopes))
    plot_curves(out, means)
