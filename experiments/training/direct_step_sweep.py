"""One-minibatch projected-step-size oracle and true-loss check."""

import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.training.mnist_qat import MLP, data_split
from experiments.training.tiny_oracle import codebooks


def main():
    out = Path("experiments/results/procedural_training/direct_step_sweep")
    out.mkdir(parents=True, exist_ok=True)
    images, labels, _, _, train_ids, _, manifest = data_split()
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    torch.manual_seed(11)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(images[train_ids[:256]].reshape(256,-1), dtype=torch.float32, device=device)/255
    y = torch.tensor(labels[train_ids[:256]].astype("int64"), device=device)
    original = MLP().to(device)
    initial = {key:value.detach().clone() for key,value in original.state_dict().items()}
    scales = [float(layer.weight.detach().square().mean().sqrt()) for layer in original.layers]
    rows=[]
    for family in ("scalar", "discovered"):
        model = MLP(codebooks()[family], scales).to(device)
        model.load_state_dict(initial, strict=False)
        with torch.no_grad():
            for layer, scale in zip(model.layers, scales):
                layer.weight.copy_(model.nearest(layer.weight, scale)[0])
        baseline = F.cross_entropy(model(x), y)
        baseline.backward()
        weights = [layer.weight.detach().clone() for layer in model.layers]
        gradients = [layer.weight.grad.detach().clone() for layer in model.layers]
        indices_before = [model.nearest(layer.weight, scale)[1] for layer, scale in zip(model.layers, scales)]
        for eta in (.1,1.,10.,100.,1000.):
            changed=total=0
            with torch.no_grad():
                for layer, scale, weight, gradient, old in zip(model.layers, scales, weights, gradients, indices_before):
                    decoded, indices = model.nearest(weight-eta*gradient, scale)
                    layer.weight.copy_(decoded)
                    changed += int((indices!=old).sum())
                    total += indices.numel()
                actual=float(F.cross_entropy(model(x), y))
            rows.append(dict(family=family, eta=eta, changed_fraction=changed/total,
                             initial_loss=float(baseline), actual_loss=actual,
                             loss_change=actual-float(baseline)))
    with (out/"metrics.csv").open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    (out/"config.json").write_text(json.dumps(dict(seed=11,batch=256,
        note="Single fixed minibatch, no LR selection from test"),indent=2)+"\n")
    print(json.dumps(rows, indent=2))


if __name__=="__main__":
    main()
