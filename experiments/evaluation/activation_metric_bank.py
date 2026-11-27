"""Frozen single-projection predictor bank for activation-aware metrics.

This tests metric prediction on an exposed SmolLM2 evaluation set.  It does
not change the encoded objective or claim fresh held-out confirmation.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
import torch
from tokenizers import Tokenizer

from experiments.evaluation.int_controls import GroupedScalar, UniformScalar
from experiments.evaluation.seedlm_model import calibration, load_model, perplexity, token_ids
from experiments.manifests.weights import PROJECTIONS, Weights


LAYERS=(0,15,29)


def write_csv(path, rows):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--layers",type=int,nargs="+",default=list(LAYERS))
    parser.add_argument("--out",type=Path,
                        default=Path("experiments/results/experiments-13-18/activation_metric_bank"))
    args=parser.parse_args()
    torch.set_num_threads(4)
    checkpoint=Path("data/smollm2-135m.safetensors")
    assets=Path("data/smollm2")
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    model=load_model(checkpoint,assets,torch.float32).to("cuda")
    source=Weights(checkpoint)
    tokenizer=Tokenizer.from_file(str(assets/"tokenizer.json"))
    calib_tokens=token_ids(assets,"validation",tokenizer)[:2048]
    test_tokens=token_ids(assets,"test",tokenizer)[:2048]
    with torch.inference_mode():
        calibration_inputs=calibration(model,calib_tokens,128,"cuda",args.layers)
        test_inputs=calibration(model,test_tokens,128,"cuda",args.layers)
        baseline,count=perplexity(model,test_tokens,128,1,"cuda")
    codecs={
        "odd8_3":UniformScalar(8,3),
        "odd8_50":UniformScalar(8,50),
        "zero8_row":GroupedScalar(8,"channel"),
        "odd4_3":UniformScalar(4,3),
        "zero4_group64":GroupedScalar(4,64),
    }
    rows=[]
    for layer in args.layers:
        for name,group in PROJECTIONS.items():
            module=getattr(getattr(model.model.layers[layer],group),name)
            old=module.weight
            w=source.get(layer,name).astype(np.float32)
            x=torch.from_numpy(calibration_inputs[(layer,name)]).to("cuda")
            x_test=torch.from_numpy(test_inputs[(layer,name)]).to("cuda")
            wt=torch.from_numpy(w).to("cuda")
            h=x.square().mean(0)
            with torch.inference_mode():
                denom_weight=float(wt.square().sum())
                denom_diag=float((wt.square().sum(0)*h).sum())
                denom_full=float((x@wt.T).square().sum())
                denom_test=float((x_test@wt.T).square().sum())
            for method,codec in codecs.items():
                begin=time.perf_counter()
                restored,payload=codec.encode_matrix(w)
                encode_seconds=time.perf_counter()-begin
                err=torch.from_numpy(w-restored).to("cuda")
                with torch.inference_mode():
                    err2=err.square().sum(0)
                    weight_nmse=float(err2.sum())/denom_weight
                    diagonal_nmse=float((err2*h).sum())/denom_diag
                    full_nmse=float((x@err.T).square().sum())/denom_full
                    heldout_output_nmse=float((x_test@err.T).square().sum())/denom_test
                    module.weight=torch.nn.Parameter(torch.from_numpy(restored).to("cuda"),requires_grad=False)
                    nll,_=perplexity(model,test_tokens,128,1,"cuda")
                    module.weight=old
                rows.append(dict(layer=layer,projection=name,codec=method,
                                 projection_bpw=len(payload)*8/w.size,
                                 weight_nmse=weight_nmse,diagonal_nmse=diagonal_nmse,
                                 full_calibration_nmse=full_nmse,
                                 heldout_output_nmse=heldout_output_nmse,
                                 baseline_nll=baseline,nll=nll,delta_nll=nll-baseline,
                                 scored_targets=count,encode_seconds=encode_seconds))
                print(layer,name,method,nll-baseline,flush=True)
                write_csv(out/"metrics.csv",rows)
    correlations=[]
    for codec in codecs:
        candidates=[row for row in rows if row["codec"]==codec]
        for metric in ("weight_nmse","diagonal_nmse","full_calibration_nmse",
                       "heldout_output_nmse"):
            statistic=spearmanr([row[metric] for row in candidates],
                                [row["delta_nll"] for row in candidates])
            correlations.append(dict(codec=codec,metric=metric,candidates=len(candidates),
                                     spearman=float(statistic.statistic),pvalue=float(statistic.pvalue)))
    write_csv(out/"correlations.csv",correlations)
    (out/"config.json").write_text(json.dumps(dict(
        layers=args.layers,projections=list(PROJECTIONS),codecs=list(codecs),
        calibration_tokens=2048,evaluation_tokens=2048,context=128,
        calibration_split="validation, already exposed in prior work",
        evaluation_split="test, already exposed in prior work",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        torch=torch.__version__,hip=torch.version.hip,
        metrics="uncentered second moment; one projection replaced at a time",
        note="Predictor-only exploratory bank; rates and candidate families differ, and no alpha/weighted encoding is fitted"),indent=2)+"\n")


if __name__=="__main__":main()
