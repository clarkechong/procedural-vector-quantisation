"""Scatter plots from the frozen SmolLM2 activation-metric bank."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root=Path("results/e13-18-activation-landscape")
    rows=[]
    for filename in ("activation_metric_bank/metrics.csv","activation_metric_layer11/metrics.csv"):
        with (root/filename).open() as handle:
            rows.extend(csv.DictReader(handle))
    colors={"odd8_3":"#d62728","odd8_50":"#ff7f0e","zero8_row":"#2ca02c",
            "odd4_3":"#9467bd","zero4_group64":"#1f77b4"}
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,metric,label in zip(axes,
            ("weight_nmse","diagonal_nmse","full_calibration_nmse"),
            ("Plain weight NMSE","Diagonal activation NMSE","Full second-moment NMSE")):
        for codec,color in colors.items():
            selected=[r for r in rows if r["codec"]==codec]
            ax.scatter([float(r[metric]) for r in selected],
                       [float(r["delta_nll"]) for r in selected],
                       s=23,alpha=.7,color=color,label=codec)
        special=next(r for r in rows if r["layer"]=="11" and r["projection"]=="down_proj"
                     and r["codec"]=="odd8_50")
        ax.annotate("layer 11 down",(float(special[metric]),float(special["delta_nll"])),
                    xytext=(8,-14),textcoords="offset points",fontsize=8,
                    arrowprops=dict(arrowstyle="-",lw=.6))
        ax.set(xlabel=label,ylabel="Single-projection ΔNLL (nats/token)")
        ax.set_xscale("symlog",linthresh=1e-5)
    axes[0].legend(fontsize=7)
    fig.suptitle("Frozen candidate bank; exposed SmolLM2 validation/test splits")
    fig.savefig(root/"activation_metric_bank/predictor_scatter.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":main()
