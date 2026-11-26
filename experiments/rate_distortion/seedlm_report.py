"""Summarize paired reconstruction results and render the SeedLM experiment."""

import csv
from collections import defaultdict
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.encoding.seedlm import Config, decode, pack, seed_ids


GROUPS = ("experiment", "block", "latent", "seed_bits", "coefficient_bits", "family",
          "bits_per_weight", "fitting", "rule", "candidates", "dataset")


def aggregate(path):
    groups = defaultdict(dict)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[field] for field in GROUPS)
            groups[key][row["layer"], row["projection"]] = float(row["nmse"])
    summaries = []
    with (path.parent / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(*GROUPS, "matrices", "mean_nmse", "sd_nmse"))
        writer.writeheader()
        for key, matrices in sorted(groups.items()):
            values = list(matrices.values())
            row = dict(zip(GROUPS, key))
            row.update(matrices=len(values), mean_nmse=float(np.mean(values)),
                       sd_nmse=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            writer.writerow(row)
            summaries.append(row)
    paired_controls(path.parent, groups)
    return summaries


def paired_controls(out, groups):
    fields = (*GROUPS[:-1], "control", "matrices", "pretrained_nmse", "control_nmse",
              "mean_control_minus_pretrained", "matrix_wins")
    with (out / "paired_controls.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, observed in sorted(groups.items()):
            if key[-1] != "pretrained":
                continue
            row = dict(zip(GROUPS[:-1], key[:-1]))
            for control in ("global_shuffle", "within_block_shuffle", "gaussian"):
                reference = groups.get((*key[:-1], control), {})
                names = observed.keys() & reference.keys()
                if not names:
                    continue
                original = np.array([observed[name] for name in names])
                randomized = np.array([reference[name] for name in names])
                difference = randomized - original
                writer.writerow({**row, "control": control, "matrices": len(names),
                                 "pretrained_nmse": original.mean(), "control_nmse": randomized.mean(),
                                 "mean_control_minus_pretrained": difference.mean(),
                                 "matrix_wins": np.count_nonzero(difference > 0)})


def benchmark_decoder(out):
    """Time the packed NumPy reference on an equal-sized stream for each codec."""
    path = out / "decode_cost.csv"
    if not path.is_file():
        return
    rng = np.random.default_rng(753)
    fields = ("family", "block", "latent", "seed_bits", "coefficient_bits", "bits_per_weight",
              "machine", "weights", "measurements", "median_ns_per_weight")
    with path.open() as metadata, (out / "cpu_decode.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in csv.DictReader(metadata):
            config = Config(*(int(row[key]) for key in ("block", "latent", "seed_bits", "coefficient_bits")),
                            family=row["family"])
            count = 49_152 // config.block
            seeds = rng.choice(seed_ids(config), count)
            limit = 1 << (config.coefficient_bits - 1)
            coefficients = rng.integers(-limit, limit, size=(count, config.latent))
            exponents = rng.integers(-8, 8, size=count)
            payload = pack(config, seeds, coefficients, exponents)
            decode(config, payload, count)
            durations = []
            for _ in range(7):
                start = time.perf_counter_ns()
                decode(config, payload, count)
                durations.append(time.perf_counter_ns() - start)
            weights = count * config.block
            values = {key: row[key] for key in fields[:6]}
            writer.writerow({**values, "machine": platform.machine(), "weights": weights,
                             "measurements": len(durations),
                             "median_ns_per_weight": float(np.median(durations) / weights)})


def design_plot(out, rows):
    screen = out / "screen.csv"
    if not screen.is_file():
        return
    with screen.open() as handle:
        points = [(float(r["bits_per_weight"]), float(r["nmse"])) for r in csv.DictReader(handle)]
    finalists = [r for r in rows if r["experiment"] == "screen_finalist"
                 and r["rule"] == "refine" and r["dataset"] == "pretrained"]
    with (out / "scalar_and_pilot_baselines.csv").open() as handle:
        alternatives = defaultdict(list)
        for row in csv.DictReader(handle):
            if row["method"].startswith(("uniform", "lloyd_max")):
                key = (float(row["bits_per_weight"]), row["block"], row["method"])
                alternatives[key].append(float(row["nmse"]))
    scalar = [min((float(np.mean(values)) for (rate, _, _), values in alternatives.items()
                   if rate <= budget), default=1) for budget in (1, 2, 3, 4)]
    fig, ax = plt.subplots(figsize=(8.5, 5), layout="constrained")
    ax.scatter(*zip(*points), s=14, color="0.55", alpha=0.33, label="Screened on separate rows")
    ax.scatter([float(r["bits_per_weight"]) for r in finalists],
               [r["mean_nmse"] for r in finalists], color="#1763a6", s=75,
               marker="D", label="Exhaustive search on held-out rows", zorder=3)
    ax.plot((1, 2, 3, 4), scalar, color="#b95c24", marker="o",
            label="Best scalar quantizer within budget")
    ax.set(title="Seeded subspace rate–distortion screen",
           xlabel="Bits per weight", ylabel="Mean NMSE", xticks=[1, 2, 3, 4], yscale="log")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(out / "design_space.png", dpi=170)
    plt.close(fig)


def report(out):
    rows = aggregate(out / "measurements.csv")
    benchmark_decoder(out)
    design_plot(out, rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    fits = {"paper": "Published exponent", "range": "Range-fitted exponent", "float": "Float coefficients"}
    for ax, block, rate in zip(axes[0], ("12", "8"), (3, 4)):
        for rule, label in fits.items():
            points = sorted(
                (int(r["candidates"]), r["mean_nmse"])
                for r in rows if r["experiment"] == "published" and r["block"] == block
                and r["rule"] == rule and r["dataset"] == "pretrained"
            )
            ax.plot(*zip(*points), marker="o", label=label)
        ax.set(title=f"{rate}-bit SeedLM geometry · {block} weights",
               xlabel="Candidate seeds searched", ylabel="Mean NMSE", xscale="log", yscale="log")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)

    reference = [r for r in rows if r["experiment"] == "published"
                 and r["rule"] == "paper" and r["candidates"] == "65535"]
    labels = {"pretrained": "Pretrained", "global_shuffle": "Global shuffle",
              "within_block_shuffle": "Within-block shuffle", "gaussian": "Gaussian"}
    for index, (dataset, label) in enumerate(labels.items()):
        selected = sorted((float(r["bits_per_weight"]), r["mean_nmse"])
                          for r in reference if r["dataset"] == dataset)
        rates, errors = zip(*selected)
        axes[1, 0].bar(np.asarray(rates) + 0.15 * (index - 1.5), errors, width=0.14, label=label)
    axes[1, 0].set(title="Published recipe across controls",
                   xlabel="Bits per weight", ylabel="Mean NMSE", xticks=[3, 4])
    axes[1, 0].legend(fontsize=8)

    variants = [r for r in rows if r["dataset"] == "pretrained" and
                ((r["experiment"] == "matched_family" and r["rule"] == "range") or
                 (r["experiment"] == "published" and r["rule"] == "range" and r["candidates"] == "65535"))]
    families = list(dict.fromkeys(r["family"] for r in variants))
    for delta, block, label in ((-0.19, "12", "3 bits"), (0.19, "8", "4 bits")):
        errors = {(r["family"]): r["mean_nmse"] for r in variants if r["block"] == block}
        axes[1, 1].bar(np.arange(len(families)) + delta, [errors.get(f, 0) for f in families],
                       width=0.37, label=label)
    axes[1, 1].set(title="Generator families with range-fitted exponent",
                   ylabel="Mean NMSE", xticks=np.arange(len(families)),
                   xticklabels=[f.replace("_", " ") for f in families])
    axes[1, 1].tick_params(axis="x", labelrotation=30)
    axes[1, 1].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_axisbelow(True)
        ax.grid(alpha=0.22, axis="y")
    fig.savefig(out / "seedlm.png", dpi=170)
    plt.close(fig)
