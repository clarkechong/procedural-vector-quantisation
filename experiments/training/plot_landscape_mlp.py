"""Plots from saved paired MLP metrics; no hard-coded experimental results."""

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--result",type=Path,default=Path("experiments/results/experiments-13-18/landscape_mlp"))
    args=parser.parse_args()
    root=args.result
    final_epoch=json.loads((root/"config.json").read_text())["epochs"]
    curves=read(root/"curves.csv")
    cross=read(root/"cross_codec.csv")
    connectivity=read(root/"connectivity.csv")+read(root/"dense_control_connectivity.csv")
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    colors={"dense":"#1f77b4","scalar2":"#ff7f0e","mul120_2bpw":"#2ca02c",
            "dense_independent":"#9467bd"}
    for method in ("dense","scalar2","mul120_2bpw"):
        grouped=defaultdict(list)
        for row in curves:
            if row["branch"]==method:
                grouped[int(row["epoch"])].append(float(row["validation_ce"]))
        epochs=sorted(grouped)
        mean=[np.mean(grouped[k]) for k in epochs]
        sd=[np.std(grouped[k],ddof=1) for k in epochs]
        axes[0,0].plot(epochs,mean,label=method,color=colors[method])
        axes[0,0].fill_between(epochs,np.asarray(mean)-sd,np.asarray(mean)+sd,
                               color=colors[method],alpha=.15)
    axes[0,0].set(xlabel="Epoch",ylabel="Exposed validation cross-entropy",
                  title="Paired scratch training, mean ± SD over 3 seeds")
    axes[0,0].set_ylim(0,.4);axes[0,0].legend()
    sources=["dense","scalar2","mul120_2bpw","release_scalar2","release_mul120_2bpw"]
    codecs=["scalar2","mul120_2bpw","scalar4"]
    grid=np.zeros((len(sources),len(codecs)))
    for i,source in enumerate(sources):
        for j,codec in enumerate(codecs):
            grid[i,j]=np.mean([float(row["delta_ce"]) for row in cross
                               if row["source"]==source and row["codec"]==codec])
    im=axes[0,1].imshow(grid,cmap="viridis",aspect="auto")
    for i in range(len(sources)):
        for j in range(len(codecs)):
            axes[0,1].text(j,i,f"{grid[i,j]:.3f}",ha="center",va="center",
                           color="white" if grid[i,j]<.7 else "black")
    axes[0,1].set_xticks(range(len(codecs)),codecs,rotation=20)
    axes[0,1].set_yticks(range(len(sources)),sources)
    axes[0,1].set(title="Independent codec transfer: mean ΔCE")
    fig.colorbar(im,ax=axes[0,1],shrink=.75)
    for method in ("dense_independent","mul120_2bpw","release_mul120_2bpw"):
        for layout,style in (("raw","--"),("aligned","-")):
            by_t=defaultdict(list)
            for row in connectivity:
                if row["method"]==method and row["layout"]==layout:
                    by_t[float(row["t"])].append(float(row["validation_ce"]))
            if not by_t:continue
            ts=sorted(by_t)
            axes[1,0].plot(ts,[np.mean(by_t[t]) for t in ts],style,
                           label=f"{method} {layout}",
                           color=colors.get(method,"#2ca02c"))
    axes[1,0].set(xlabel="Interpolation t from paired dense endpoint",
                  ylabel="Exposed validation CE",title="Dense-space linear connectivity")
    axes[1,0].legend(fontsize=8)
    for branch in ("release_dense","release_scalar2","release_mul120_2bpw"):
        by_epoch=defaultdict(list)
        for row in curves:
            if row["branch"]==branch:
                by_epoch[int(row["epoch"])].append(float(row["validation_ce"]))
        epochs=sorted(by_epoch)
        axes[1,1].plot([e-final_epoch for e in epochs],
                       [np.mean(by_epoch[e]) for e in epochs],label=branch)
    axes[1,1].set(xlabel="Dense continuation epochs after release",
                  ylabel="Exposed validation CE",title="Remove constraint and continue dense training")
    axes[1,1].legend(fontsize=8)
    fig.savefig(root/"landscape_summary.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":main()
