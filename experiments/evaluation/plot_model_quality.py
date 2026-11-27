"""Plot completed SmolLM2 full-model NMSE and WikiText-2 perplexity measurements."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
COLORS = {"Synthesized": "#3567BC", "SeedLM": "#208978", "Scalar": "#BD731D", "Original": "#303844"}


def measurements(laptop, mi300x=None):
    laptop = laptop.resolve()
    if mi300x is not None:
        mi300x = mi300x.resolve()
    rows = []
    labels = {
        "original": ("Original", "Original", "*"),
        "synthesized_2": ("Synthesized 2b", "Synthesized", "o"),
        "synthesized_2p5": ("Synthesized 2.5b", "Synthesized", "o"),
        "synthesized_3": ("Synthesized 3b", "Synthesized", "o"),
        "synthesized_3p5": ("Synthesized 3.5b", "Synthesized", "o"),
        "seedlm_all_seeds_3": ("SeedLM all seeds 3b", "SeedLM", "^"),
        "seedlm_all_seeds_4": ("SeedLM all seeds 4b", "SeedLM", "^"),
        "seedlm_paper_3": ("SeedLM shortlist 3b", "SeedLM", "s"),
        "seedlm_paper_4": ("SeedLM shortlist 4b", "SeedLM", "s"),
    }
    for directory, run_name in ((laptop, "Laptop CUDA"), (mi300x, "MI300X ROCm")):
        if directory is None:
            continue
        for method, (label, family, marker) in labels.items():
            source = directory / method / "perplexity.csv"
            if not source.exists():
                continue
            with source.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["method"] != method or int(row["scored_tokens"]) != 304680:
                        continue
                    rows.append(dict(method=method, label=label, family=family, marker=marker,
                                     bits_per_weight=float(row["projection_bpw"]),
                                     weight_nmse=float(row["weight_nmse"]), perplexity=float(row["perplexity"]),
                                     run=run_name, source=str(source.relative_to(ROOT))))
    source = ROOT / "results/s03-computational-codebooks/model_quality.csv"
    historical = {
        "int4_group64": ("Group-64 INT4 4.25b", "Scalar", "D"),
        "indexed_refine": ("SeedLM indexed refinement 4b", "SeedLM", "D"),
        "refine": ("SeedLM fixed-255 refined 4b", "SeedLM", "v"),
        "paper": ("SeedLM fixed-255 paper 4b", "SeedLM", "v"),
    }
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] not in historical or int(row["scored_tokens"]) != 304680:
                continue
            label, family, marker = historical[row["method"]]
            rows.append(dict(method=row["method"], label=label, family=family, marker=marker,
                             bits_per_weight=float(row["projection_bpw"]),
                             weight_nmse=float(row["weight_nmse"]), perplexity=float(row["perplexity"]),
                             run="Earlier CPU", source=str(source.relative_to(ROOT))))
    return rows


def plot(rows, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#303844", "text.color": "#303844",
                         "xtick.color": "#56606E", "ytick.color": "#56606E"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=.09, right=.97, top=.78, bottom=.23, wspace=.20)
    fig.suptitle("Weight reconstruction error and language-model quality", x=.075, y=.96,
                 ha="left", fontsize=19, fontweight="bold")
    fig.text(.075, .905, "SmolLM2-135M · WikiText-2 · 304,680 scored tokens · lower is better on both axes",
             fontsize=11, color="#56606E")
    synthesized = sorted((row for row in rows if row["family"] == "Synthesized"),
                         key=lambda row: row["weight_nmse"])
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(axis="both", which="major", alpha=.22, linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("Full projection weight NMSE", labelpad=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.tick_params(which="minor", left=False)
        ax.plot([row["weight_nmse"] for row in synthesized],
                [row["perplexity"] for row in synthesized], color=COLORS["Synthesized"],
                linewidth=1.5, linestyle="--", alpha=.65)
        for row in rows:
            historical = row["run"] == "Earlier CPU"
            color = COLORS[row["family"]]
            ax.scatter(row["weight_nmse"], row["perplexity"], marker=row["marker"],
                       s=115 if row["family"] == "Original" else 65,
                       facecolor="white" if historical else color, edgecolor=color,
                       linewidth=1.5, zorder=4)
    axes[0].set(xlim=(-.005, .233), ylim=(10, 1e7), ylabel="Test perplexity (logarithmic scale)")
    axes[0].set_title("All completed full-model measurements", loc="left", fontsize=12, pad=12)
    axes[0].xaxis.set_major_locator(MultipleLocator(.05))
    axes[1].set(xlim=(-.003, .075), ylim=(10, 3000))
    axes[1].set_title("Detail: lower reconstruction error", loc="left", fontsize=12, pad=12)
    axes[1].xaxis.set_major_locator(MultipleLocator(.02))
    main_offsets = {"synthesized_2": (-12, -24), "synthesized_2p5": (14, -23),
                    "seedlm_paper_3": (-18, 23), "seedlm_all_seeds_3": (7, 12),
                    "refine": (-12, -28), "paper": (-8, 15)}
    detail_offsets = {"original": (9, 8), "int4_group64": (8, -18),
                      "indexed_refine": (8, 8), "seedlm_all_seeds_4": (-12, 8),
                      "seedlm_paper_4": (8, 9), "synthesized_3": (-12, 9)}
    for ax, offsets in zip(axes, (main_offsets, detail_offsets)):
        for row in rows:
            if row["method"] not in offsets:
                continue
            dx, dy = offsets[row["method"]]
            value = f"{row['perplexity']:,.0f}" if row["perplexity"] >= 1000 else f"{row['perplexity']:.2f}"
            ax.annotate(f"{row['label']}\n{value}", (row["weight_nmse"], row["perplexity"]),
                        xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
                        ha="right" if dx < 0 else "left", va="bottom" if dy > 0 else "top",
                        arrowprops=dict(arrowstyle="-", color="#A0A7B0", lw=.7), zorder=5)
    handles = [Line2D([], [], marker="o" if family != "Scalar" else "D", linestyle="none",
                      color=color, label=family, markersize=6)
               for family, color in COLORS.items() if family != "Original"]
    handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                          markeredgecolor="#56606E", label="Hollow: earlier CPU run", markersize=6))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.07, .105), ncol=4, frameon=False)
    fig.text(.075, .075, "Dashed line joins measured points for the same frozen synthesized program; it is not a fitted law.",
             fontsize=9, color="#56606E")
    fig.text(.075, .045, "All runs compress the same 210 projections. Different methods and hardware are pooled; NMSE alone does not determine perplexity.",
             fontsize=9, color="#56606E")
    fig.savefig(out / "nmse_perplexity.png", dpi=180, facecolor="white")
    fig.savefig(out / "nmse_perplexity.svg", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        if row["family"] == "Original":
            continue
        color = COLORS[row["family"]]
        ax.scatter(row["bits_per_weight"], row["weight_nmse"], color=color, marker=row["marker"])
        if row["method"] == "synthesized_3p5":
            ax.annotate("Synthesized 3.5b (measured)",
                        (row["bits_per_weight"], row["weight_nmse"]), xytext=(8, 8),
                        textcoords="offset points")
    ax.set(xlabel="Projection bits per weight", ylabel="Full projection weight NMSE",
           title="Measured SmolLM2-135M compression results")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "bitrate_nmse.png", dpi=180, facecolor="white")
    fig.savefig(out / "bitrate_nmse.svg", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/laptop-rtx3050")
    parser.add_argument("--mi300x", type=Path, default=ROOT / "results/s06-hardware-evaluation/mi300x-3p5")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/laptop-rtx3050/plots")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = measurements(args.runs, args.mi300x)
    with (args.out / "nmse_perplexity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["weight_nmse"]))
    plot(rows, args.out)
    print(f"Plotted {len(rows)} full-model measurements in {args.out}")


if __name__ == "__main__":
    main()
