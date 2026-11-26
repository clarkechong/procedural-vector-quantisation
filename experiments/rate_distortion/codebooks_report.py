"""Summarize the computational codebook experiment and generate static figures."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.encoding.seedlm import Config, basis, seed_ids


def read(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def grouped(rows, keys, field="nmse"):
    values = defaultdict(list)
    for row in rows:
        values[tuple(row[key] for key in keys)].append(float(row[field]))
    return {key: float(np.mean(metric)) for key, metric in values.items()}


def beta_minimum(counts, block=8, latent=3):
    """Integrate the survival function of the smallest IID beta residual."""
    alpha, beta = (block - latent) / 2, latent / 2
    nodes, weights = np.polynomial.legendre.leggauss(128)
    nodes, weights = (nodes + 1) / 2, weights / 2
    x = nodes[:, None] * nodes
    normalizer = math.gamma(alpha + beta) / (math.gamma(alpha) * math.gamma(beta))
    cdf = normalizer * nodes * ((x ** (alpha - 1) * (1 - x) ** (beta - 1)) @ weights)
    return (np.exp(np.log1p(-np.clip(cdf, 0, 1 - np.finfo(float).eps)) *
                   np.asarray(counts)[:, None])) @ weights


def bootstrap_difference(first, second):
    """Paired matrix resampling, conditional on the measured checkpoint."""
    diffs = np.asarray(first) - np.asarray(second)
    rng = np.random.default_rng(94)
    draws = rng.integers(0, len(diffs), size=(10000, len(diffs)))
    return np.quantile(diffs[draws].mean(axis=1), (0.025, 0.975))


def geometry(out, families, curves):
    with (out / "geometry.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("family", "count", "floating_nmse", "iid_beta_expected_min_nmse",
                         "unique_bases", "distinct_projectors_rounded_1e6", "mean_rank"))
        counts = (1, 15, 255, 4095, 65535)
        ideal = beta_minimum(counts)
        for family in families:
            config = Config(8, 3, family=family)
            bases = basis(config, seed_ids(config)).astype(np.float64)
            matrices = bases @ np.linalg.pinv(bases, rcond=1e-13)
            triangular = np.triu_indices(8)
            projectors = np.rint(matrices[:, triangular[0], triangular[1]] * 1e6).astype(np.int32)
            distinct = len(np.unique(projectors, axis=0))
            unique = len(np.unique(bases.reshape(len(bases), -1), axis=0))
            rank = np.trace(matrices, axis1=1, axis2=2).mean()
            for count, expectation in zip(counts, ideal):
                writer.writerow((family, count, curves[family, "float", str(count)], expectation,
                                 unique, distinct, rank))


def frontier(out, rates, curves, decode):
    times = {row["family"]: float(row["gweights_per_s"]) for row in decode
             if row["operation"] == "materialize"}
    basis_multiply = {"lfsr": 3, "binary": 0, "ternary": 0, "xorshift": 3, "splitmix": 3}
    candidates = [(family, 4., curves[family, "refine", "65535"], speed)
                  for family, speed in times.items()]
    with (out / "pareto_cpu.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("family", "bpw", "nmse", "decode_gweights_per_s",
                         "basis_multiplications_per_weight", "pareto_on_m1"))
        for family, bpw, nmse, speed in candidates:
            dominated = any(r <= bpw and d <= nmse and s >= speed and
                            (r < bpw or d < nmse or s > speed) for _, r, d, s in candidates)
            writer.writerow((family, bpw, nmse, speed, basis_multiply[family], int(not dominated)))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    counts = np.array([1, 15, 255, 4095, 65535])
    families = ("lfsr", "binary", "ternary", "xorshift", "splitmix", "hadamard")
    palette = dict(zip(families, plt.cm.tab10.colors))
    for family in families:
        axes[0].loglog(counts, [curves[family, "float", str(c)] for c in counts],
                       marker="o", label=family, color=palette[family])
    axes[0].loglog(counts, beta_minimum(counts), "--", color="black", label="IID random subspaces")
    axes[0].set(xlabel="Seeds searched", ylabel="Floating projection NMSE",
                title="Codebook coverage as search grows")
    axes[0].legend(frameon=False, fontsize=8)
    for family, _, nmse, speed in candidates:
        axes[1].scatter(speed, nmse, color=palette[family])
        offset = (4, -10) if family == "xorshift" else (4, 4)
        axes[1].annotate(family, (speed, nmse), xytext=offset, textcoords="offset points", fontsize=9)
    axes[1].set(xlabel="Single-core Apple M1 decode (Gweights/s)", ylabel="Refined 4-bit NMSE",
                title="Packed decoder trade-off")
    axes[1].margins(x=0.21, y=0.15)
    for ax in axes:
        ax.grid(alpha=0.25)
    figure.savefig(out / "codebook_coverage.png", dpi=180)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    series = (
        ("Direct seed (C=8, C=6)", (("direct_c8", "3"), ("direct_c6", "4"))),
        ("Additive seeds", (("additive", "4"),)),
        ("Signed permutation", (("signed_permutation", "4"),)),
        ("E8 and residual E8, rotated", (("e8_shaped_rht64", "2"), ("e8_residual_rht64", "4"))),
        ("1MAD trellis", (("trellis_1mad", "3"), ("trellis_1mad", "4"))),
        ("1MAD trellis, rotated", (("trellis_1mad_rht64", "3"), ("trellis_1mad_rht64", "4"))),
    )
    for label, points in series:
        ax.semilogy([float(p[1]) for p in points], [rates[p] for p in points], "o-", label=label)
    ax.semilogy([4], [curves["lfsr", "refine", "65535"]], "^", label="Refined dense LFSR")
    x = np.linspace(2, 4, 100)
    ax.semilogy(x, np.exp2(-2 * x), "k--", label="IID Gaussian asymptote")
    ax.set(xlabel="Payload bits per weight", ylabel="Held-out NMSE", title="Vector and trellis baselines")
    ax.set_xticks((2, 3, 4))
    ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    figure.savefig(out / "rate_distortion.png", dpi=180)
    plt.close(figure)


def model_summary(out):
    results, activations = [], []
    for directory in ("model", "model-indexed"):
        model = out / directory
        if (model / "perplexity.csv").is_file():
            results.extend(read(model / "perplexity.csv"))
            activations.extend(read(model / "activation_error.csv"))
    if not results:
        return
    weight_means = grouped(activations, ("method",), "weight_nmse")
    activation_means = grouped(activations, ("method",), "output_nmse")
    fields = ("method", "projection_bpw", "weight_nmse", "sampled_mean_weight_nmse",
              "sampled_mean_activation_nmse", "perplexity", "scored_tokens")
    with (out / "model_quality.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            record = {field: row[field] for field in fields if field in row}
            key = (row["method"],)
            record["sampled_mean_weight_nmse"] = weight_means.get(key, 0)
            record["sampled_mean_activation_nmse"] = activation_means.get(key, 0)
            writer.writerow(record)


def summarize(out):
    subspace = read(out / "subspaces.csv")
    alternatives = read(out / "other_codebooks.csv") + read(out / "rotated_baselines.csv")
    curves = grouped(subspace, ("family", "rule", "candidates"))
    rates = grouped(alternatives, ("method", "payload_bpw"))
    oracle_rows = read(out / "oracle.csv")
    fields = ("paper_sse", "refine_sse", "bounded_sse", "uncapped_sse", "free_scale_sse", "float_sse")
    finalists = ("lfsr", "ternary")
    for family in ("lfsr", "binary", "ternary"):
        entries = [r for r in oracle_rows if r["family"] == family]
        weighted = []
        for field in fields:
            per_matrix = defaultdict(lambda: [0., 0.])
            for row in entries:
                sums = per_matrix[row["layer"], row["projection"]]
                sums[0] += float(row[field])
                sums[1] += float(row["energy"])
            weighted.append(float(np.mean([err / energy for err, energy in per_matrix.values()])))
        print("Oracle:", family, dict(zip(fields, weighted)),
              "visited median/max", np.median([int(r["visited_seeds"]) for r in entries]),
              max(int(r["visited_seeds"]) for r in entries))
    for key, value in sorted(rates.items()):
        print("Vector/trellis:", key, value)
    for other in ("ternary", "binary", "xorshift", "splitmix"):
        ref, alt = [
            [float(r["nmse"]) for r in subspace
             if (r["family"], r["rule"], r["candidates"]) == (family, "refine", "65535")]
            for family in ("lfsr", other)]
        print("Paired NMSE delta vs LFSR:", other, np.mean(np.asarray(alt) - ref),
              bootstrap_difference(alt, ref))
    for family in finalists:
        for rule in ("paper", "refine", "float"):
            print("Subspace:", family, rule, [curves[family, rule, str(c)] for c in (1, 15, 255, 4095, 65535)])
    geometry(out, ("lfsr", "binary", "ternary", "xorshift", "splitmix", "hadamard"), curves)
    frontier(out, rates, curves, read(out / "cpu_decode.csv"))
    model_summary(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    summarize(parser.parse_args().out)
