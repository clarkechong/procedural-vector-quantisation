"""Plot fresh-row rate and distortion without mixing unequal test subsets."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path("results/e03-trellis")
    data = json.loads((out / "results.json").read_text())
    records = data["rate_distortion"]
    controls = data["matched_scalar_controls"]
    lengths = np.asarray(sorted({row["length"] for row in records}))
    row_counts = {length: next(row["test_rows"] for row in records
                               if row["length"] == length and row["method"] == "discovered_120")
                  for length in lengths}
    series = {}
    for name, label, color in (("discovered_120", "window, k=1", "#2864b4"),
                               ("discovered_120_stride2", "window, k=2", "#0d9279")):
        series[name] = dict(label=label, color=color,
                            nmse=[next(row["test_nmse"] for row in records
                                       if row["length"] == n and row["method"] == name)
                                  for n in lengths],
                            bpw=[next(row["payload_bpw"] for row in records
                                      if row["length"] == n and row["method"] == name)
                                 for n in lengths])
    for name, label, color, bpw in (("scalar_2_levels", "2-level scalar", "#bf7a29", 1),
                                    ("scalar_4_levels", "4-level scalar", "#b34367", 2)):
        series[name] = dict(label=label, color=color,
                            nmse=[next(row["test_nmse"] for row in controls
                                       if row["length"] == n and row["rows"] == row_counts[n]
                                       and row["method"] == name) for n in lengths],
                            bpw=[bpw] * len(lengths))
    fig, (quality, rate) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                        gridspec_kw={"height_ratios": (2, 1)})
    for item in series.values():
        quality.plot(lengths, item["nmse"], marker="o", linewidth=1.8,
                     label=item["label"], color=item["color"])
        rate.plot(lengths, item["bpw"], marker="o", linewidth=1.8,
                  color=item["color"])
    quality.set_ylabel("Test weight NMSE ↓")
    quality.set_ylim(bottom=0)
    quality.grid(alpha=0.25)
    quality.legend(ncol=2, frameon=False, loc="upper left")
    rate.set_ylabel("Payload bits / weight")
    rate.set_xlabel("Weights per row segment")
    rate.grid(alpha=0.25)
    rate.set_xscale("log", base=2)
    rate.set_xticks(lengths)
    rate.set_xticklabels([str(x) for x in lengths])
    fig.suptitle("Overlapping windows on fresh SmolLM2 down-projection rows")
    fig.text(0.11, 0.012, "Calibration/test rows are disjoint; lengths may reuse rows. 8–64 test rows per length.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(out / "rate_distortion.png", dpi=180)
    fig.savefig(out / "rate_distortion.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
