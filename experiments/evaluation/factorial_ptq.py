"""A/B/C/D activation RMS transformation pilot on one SmolLM2 projection.

Both maps have exactly 256 four-weight codewords (2 payload bpw).  Assignment
is exact for the declared diagonal weighted SSE at a fixed global scale;
three scale/assignment alternations are a heuristic joint fit.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from experiments.evaluation.seedlm_model import calibration, load_model, perplexity, token_ids
from experiments.manifests.weights import PROJECTIONS, Weights
from experiments.training.low_rate_qat import make_book


def books():
    levels=np.asarray([-3,-1,1,3],dtype=np.float32)/np.sqrt(np.float32(5))
    a=np.indices((4,4,4,4)).reshape(4,-1).T
    return {"scalar2":levels[a],"mul120":make_book("mul120",4)}


@torch.inference_mode()
def encode(weight,importance,book,scale_vector,condition,iterations=3):
    transformed=condition in ("B","D")
    weighted=condition in ("C","D")
    target=weight*scale_vector[None,:] if transformed else weight
    penalty=(importance/scale_vector.square()) if condition=="D" else importance
    if not weighted:
        penalty=torch.ones_like(penalty)
    tile_target=target.reshape(-1,4)
    tile_penalty=penalty[None,:].expand(weight.shape[0],-1).reshape(-1,4)
    book_square=book.square()
    initial=torch.sqrt((tile_penalty*tile_target.square()).sum()/tile_penalty.sum()/book_square.mean())
    scale=float(initial)
    for _ in range(iterations):
        cross=(tile_target*tile_penalty)@book.T
        norms=tile_penalty@book_square.T
        code=(scale*scale*norms-2*scale*cross).argmin(-1)
        selected=book[code]
        scale=max(float((tile_penalty*tile_target*selected).sum() /
                        (tile_penalty*selected.square()).sum()),1e-12)
    cross=(tile_target*tile_penalty)@book.T
    norms=tile_penalty@book_square.T
    code=(scale*scale*norms-2*scale*cross).argmin(-1)
    reconstructed=book[code].reshape_as(weight)*scale
    if transformed:
        reconstructed=reconstructed/scale_vector[None,:]
    objective=float((tile_penalty*(tile_target-book[code]*scale).reshape(-1,4).square()).sum())
    return reconstructed,code,scale,objective


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--layer",type=int,default=11)
    parser.add_argument("--projection",choices=list(PROJECTIONS),default="down_proj")
    parser.add_argument("--out",type=Path,
                        default=Path("experiments/results/experiments-13-18/factorial_ptq"))
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    checkpoint=Path("data/smollm2-135m.safetensors")
    assets=Path("data/smollm2")
    model=load_model(checkpoint,assets,torch.float32).to("cuda")
    tokens=Tokenizer.from_file(str(assets/"tokenizer.json"))
    calib=token_ids(assets,"validation",tokens)[:2048]
    test=token_ids(assets,"test",tokens)[:2048]
    with torch.inference_mode():
        activations=calibration(model,calib,128,"cuda",[args.layer])[(args.layer,args.projection)]
        baseline,count=perplexity(model,test,128,1,"cuda")
    x=torch.from_numpy(activations).to("cuda")
    h=x.square().mean(0).clamp_min(1e-12)
    rms=h.sqrt()
    geometric=torch.exp(rms.log().mean())
    weight=torch.from_numpy(Weights(checkpoint).get(args.layer,args.projection).astype(np.float32)).to("cuda")
    module=getattr(getattr(model.model.layers[args.layer],PROJECTIONS[args.projection]),args.projection)
    original=module.weight
    book_map={key:torch.from_numpy(value).to("cuda") for key,value in books().items()}
    rows=[];fields={}
    for family,book in book_map.items():
        for alpha in (-0.5,0.,0.25,0.5,0.75,1.):
            scale_vector=(rms/geometric).pow(alpha).clamp(1/16,16)
            clipped=int(((rms/geometric).pow(alpha)!=scale_vector).sum())
            for condition in ("A","B","C","D"):
                start=time.perf_counter()
                restored,code,scale,objective=encode(weight,h,book,scale_vector,condition)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                diff=weight-restored
                with torch.inference_mode():
                    weight_nmse=float(diff.square().sum()/weight.square().sum())
                    diag_nmse=float((diff.square().sum(0)*h).sum() /
                                    (weight.square().sum(0)*h).sum())
                    full_nmse=float((x@diff.T).square().sum()/(x@weight.T).square().sum())
                    module.weight=torch.nn.Parameter(restored,requires_grad=False)
                    nll,_=perplexity(model,test,128,1,"cuda")
                    module.weight=original
                metadata=4+(weight.shape[1]*4 if condition in ("B","D") else 0)
                payload_bytes=int(code.numel())+metadata
                row=dict(family=family,alpha=alpha,condition=condition,
                         scale=scale,clipped_scale_channels=clipped,
                         projection_bpw=payload_bytes*8/weight.numel(),
                         weight_nmse=weight_nmse,diagonal_nmse=diag_nmse,
                         full_calibration_nmse=full_nmse,objective=objective,
                         nll=nll,delta_nll=nll-baseline,
                         scored_targets=count,encode_seconds=elapsed)
                rows.append(row)
                fields[(family,alpha,condition)]=(code.cpu().numpy(),restored.cpu().numpy())
                print(family,alpha,condition,nll-baseline,flush=True)
    # Exact identities for alpha 0; alpha 1 B/D when no channel was clamped.
    checks=[]
    for family in book_map:
        for left,right,alpha in (("A","B",0.),("C","D",0.),("B","D",1.)):
            codes_left,decoded_left=fields[(family,alpha,left)]
            codes_right,decoded_right=fields[(family,alpha,right)]
            checks.append(dict(family=family,alpha=alpha,left=left,right=right,
                               code_fraction_equal=float(np.mean(codes_left==codes_right)),
                               max_weight_difference=float(np.max(np.abs(decoded_left-decoded_right)))))
    with (args.out/"metrics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (args.out/"identities.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
    (args.out/"config.json").write_text(json.dumps(dict(
        layer=args.layer,projection=args.projection,alpha_grid=[-0.5,0,0.25,0.5,0.75,1],
        conditions=["A","B","C","D"],maps=list(book_map),
        calibration_tokens=2048,evaluation_tokens=2048,context=128,
        calibration="dense inputs, exposed validation split",
        evaluation="exposed test split; single-projection replacement",
        rms_floor=1e-12,scale_clamp=[1/16,16],scale_iterations=3,
        representation_metadata="FP32 per-input S stored explicitly for B/D; execution uses equivalent original-domain decoded weight, transform runtime not measured",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        torch=torch.__version__,hip=torch.version.hip),indent=2)+"\n")


if __name__=="__main__":main()
