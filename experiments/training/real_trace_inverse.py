"""Held-out MNIST gradient traces for Experiment 07 inverse proposals."""

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.training.learned_inverse import exact_cost
from experiments.training.mnist_qat import MLP, data_split
from experiments.training.tiny_oracle import codebooks


OUT=Path("results/e04-07-reference-optimization/learned-optimizer/real-trace")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(41)
    images,labels,_,_,train_ids,_,manifest=data_split()
    (OUT/"manifest.json").write_text(json.dumps(dict(**manifest,trace_run_seed=41,
        trace_train_batches=25,trace_tiles_per_layer_per_batch=100,
        note="Independent training run; inverse trained only on synthetic targets"),indent=2)+"\n")
    device="cuda" if torch.cuda.is_available() else "cpu"
    x=torch.tensor(images[train_ids].reshape(-1,784),dtype=torch.float32,device=device)/255
    y=torch.tensor(labels[train_ids].astype("int64"),device=device)
    book=codebooks()["discovered"].astype(np.float32)
    initial=MLP().to(device)
    scales=[float(layer.weight.detach().square().mean().sqrt()) for layer in initial.layers]
    model=MLP(book,scales).to(device)
    model.load_state_dict(initial.state_dict(),strict=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    inverse=torch.nn.Sequential(torch.nn.Linear(4,128),torch.nn.ReLU(),
        torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,256)).to(device)
    inverse.load_state_dict(torch.load("results/e04-07-reference-optimization/learned-optimizer/inverse-pilot/model.pt",
        map_location=device,weights_only=True))
    inverse.eval()
    rng=np.random.default_rng(411)
    order=torch.randperm(len(x),generator=torch.Generator().manual_seed(41))[:25*256].to(device)
    records=[]
    start=time.perf_counter()
    for step,chosen in enumerate(order.split(256)):
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[chosen]),y[chosen])
        loss.backward()
        for layer,scale in zip(model.layers,scales):
            with torch.no_grad():
                _,indices=model.nearest(layer.weight,scale)
            selected=rng.choice(len(indices),min(100,len(indices)),replace=False)
            current=indices[selected].cpu().numpy().astype(np.int64)
            gradient=layer.weight.grad.detach().reshape(-1,4)[selected].cpu().numpy()
            target=book[current]-10*gradient/scale
            costs=exact_cost(target,book)
            exact=costs.min(axis=1)
            current_cost=costs[np.arange(len(current)),current]
            with torch.no_grad():
                logits=inverse(torch.tensor(target,dtype=torch.float32,device=device)).cpu().numpy()
            random=rng.integers(0,256,size=(len(current),16))
            flips=current[:,None]^(1<<np.arange(8))[None]
            deltas=book[flips]-book[current,None]
            linear_scores=np.sum((book[current]-target)[:,None,:]*deltas,axis=2)
            best_bit=linear_scores.argmin(axis=1)
            linear_choice=np.where(linear_scores[np.arange(len(current)),best_bit]<0,
                                   flips[np.arange(len(current)),best_bit],current)
            candidates={"current":current[:,None],"random16":np.column_stack((current,random)),
                "one_bit":np.column_stack((current,flips)),
                "finite_difference_linear":np.column_stack((current,linear_choice))}
            for k in (1,4,16):
                top=np.argpartition(-logits,k-1,axis=1)[:,:k]
                candidates[f"learned_top{k}"]=np.column_stack((current,top))
            nontrivial=exact<current_cost-1e-8
            for name,proposal in candidates.items():
                achieved=costs[np.arange(len(current))[:,None],proposal].min(axis=1)
                records.append(dict(step=step,layer_in_features=layer.in_features,method=name,
                    count=len(current),nontrivial=int(nontrivial.sum()),
                    regret_sum=float(np.sum(achieved-exact)),
                    hit_count=int(np.isclose(achieved,exact,atol=1e-6).sum()),
                    nontrivial_hit=int(np.isclose(achieved[nontrivial],exact[nontrivial],atol=1e-6).sum()),
                    improvement_captured_sum=float(np.sum((current_cost[nontrivial]-achieved[nontrivial])/
                                                        (current_cost[nontrivial]-exact[nontrivial]))),
                    candidates=proposal.shape[1]))
        optimizer.step()
    with (OUT/"raw_metrics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    metrics=[]
    for name in sorted(set(r["method"] for r in records)):
        group=[r for r in records if r["method"]==name]
        count=sum(r["count"] for r in group); nontrivial=sum(r["nontrivial"] for r in group)
        metrics.append(dict(method=name,count=count,nontrivial=nontrivial,
            mean_regret=sum(r["regret_sum"] for r in group)/count,
            oracle_hit=sum(r["hit_count"] for r in group)/count,
            nontrivial_oracle_hit=sum(r["nontrivial_hit"] for r in group)/nontrivial,
            improvement_captured=sum(r["improvement_captured_sum"] for r in group)/nontrivial,
            candidates=group[0]["candidates"]))
    with (OUT/"metrics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=metrics[0].keys());writer.writeheader();writer.writerows(metrics)
    (OUT/"config.json").write_text(json.dumps(dict(trace_seed=41,steps=25,tiles_per_layer=100,
        target_step_size=10,training_optimizer="shadow Adam 1e-3",device=device,
        elapsed_seconds=time.perf_counter()-start,
        inverse_source="results/e04-07-reference-optimization/learned-optimizer/inverse-pilot/model.pt"),indent=2)+"\n")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":main()
