"""One full-network update: exact versus learned local projections and task loss."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.training.mnist_qat import MLP, data_split
from experiments.training.tiny_oracle import codebooks


def main():
    out=Path("experiments/results/learned_optimizer/task_update")
    out.mkdir(parents=True,exist_ok=True)
    images,labels,_,_,train_ids,_,manifest=data_split()
    (out/"manifest.json").write_text(json.dumps(dict(**manifest,seed=41,
       batch_indices="first and second batches of seeded permutation",step_size=10),indent=2)+"\n")
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(41)
    x=torch.tensor(images[train_ids].reshape(-1,784),dtype=torch.float32,device=device)/255
    y=torch.tensor(labels[train_ids].astype("int64"),device=device)
    order=torch.randperm(len(x),generator=torch.Generator().manual_seed(41))[:512].to(device)
    train,holdout=order[:256],order[256:]
    initial=MLP().to(device)
    scales=[float(layer.weight.detach().square().mean().sqrt()) for layer in initial.layers]
    book=codebooks()["discovered"].astype(np.float32)
    model=MLP(book,scales).to(device)
    model.load_state_dict(initial.state_dict(),strict=False)
    inverse=torch.nn.Sequential(torch.nn.Linear(4,128),torch.nn.ReLU(),
        torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,256)).to(device)
    inverse.load_state_dict(torch.load("experiments/results/learned_optimizer/inverse_pilot/model.pt",
        map_location=device,weights_only=True));inverse.eval()
    train_loss=F.cross_entropy(model(x[train]),y[train]);train_loss.backward()
    with torch.no_grad():
        baseline_holdout=float(F.cross_entropy(model(x[holdout]),y[holdout]))
    original=[layer.weight.detach().clone() for layer in model.layers]
    proposal={name:[] for name in ("exact","learned_top4","one_bit")}
    movement={name:0 for name in proposal}
    rng=np.random.default_rng(441)
    for layer,scale in zip(model.layers,scales):
        with torch.no_grad():_,current=model.nearest(layer.weight,scale)
        target=(model.book[current].reshape(-1,4)-10*layer.weight.grad.detach().reshape(-1,4)/scale)
        values=target.cpu().numpy()
        candidates=torch.tensor(book,device=device)
        with torch.no_grad():
            # Chunk to keep the distance matrix bounded.
            best=[]; learned=[]; flip=[]
            for block,old in zip(target.split(4096),current.split(4096)):
                costs=(block**2).sum(dim=1,keepdim=True)+(candidates**2).sum(dim=1)[None]-2*block@candidates.T
                exact=costs.argmin(dim=1)
                logits=inverse(block)
                top=logits.topk(4,dim=1).indices
                chosen=torch.cat((old[:,None],top),dim=1)
                local=costs.gather(1,chosen)
                learned.append(chosen.gather(1,local.argmin(dim=1)[:,None]).flatten())
                flips=old[:,None]^(1<<torch.arange(8,device=device))[None]
                flipset=torch.cat((old[:,None],flips),dim=1)
                flipcost=costs.gather(1,flipset)
                flip.append(flipset.gather(1,flipcost.argmin(dim=1)[:,None]).flatten())
                best.append(exact)
            for name,ids in (("exact",torch.cat(best)),("learned_top4",torch.cat(learned)),
                             ("one_bit",torch.cat(flip))):
                proposal[name].append(model.book[ids].reshape(layer.weight.shape)*scale)
                movement[name]+=int((ids!=current).sum())
    result=dict(initial_train_loss=float(train_loss),initial_holdout_loss=baseline_holdout,
       total_tiles=sum(layer.weight.numel()//4 for layer in model.layers))
    with torch.no_grad():
        for name,matrices in proposal.items():
            for layer,values in zip(model.layers,matrices):layer.weight.copy_(values)
            result[name]=dict(train_loss=float(F.cross_entropy(model(x[train]),y[train])),
                              holdout_loss=float(F.cross_entropy(model(x[holdout]),y[holdout])),
                              changed_tiles=movement[name])
            for layer,values in zip(model.layers,original):layer.weight.copy_(values)
    result["learned_exact_identical"]=all(torch.equal(a,b) for a,b in zip(proposal["exact"],proposal["learned_top4"]))
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
