"""Static validation-learning curves from the three-seed training CSVs."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT=Path("experiments/results")
OUT=ROOT/"experiments-04-07"


def main():
    fig, axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for axis,(name,title) in zip(axes,(("mnist_qat","MNIST"),("fashion_qat","Fashion-MNIST"))):
        with (ROOT/"procedural_training"/name/"curves.csv").open() as handle:
            rows=list(csv.DictReader(handle))
        for method,label,color in (("dense","Dense","#1f77b4"),("scalar_qat","Scalar QAT","#ff7f0e"),
                                   ("procedural_qat","Procedural QAT","#2ca02c")):
            values=np.array([[float(row["validation_accuracy"]) for row in rows
                if row["method"]==method and int(row["seed"])==seed] for seed in (11,17,23)])
            epoch=np.arange(values.shape[1])
            mean=values.mean(axis=0)
            sd=values.std(axis=0,ddof=1)
            axis.plot(epoch,mean,label=label,color=color)
            axis.fill_between(epoch,mean-sd,mean+sd,color=color,alpha=.15)
        axis.set(title=title,xlabel="Epoch",xlim=(1,20),ylim=(.75,1.0))
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Validation accuracy")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT/"qat_learning_curves.png",dpi=180)
    fig.savefig(OUT/"qat_learning_curves.svg")


if __name__=="__main__":main()
