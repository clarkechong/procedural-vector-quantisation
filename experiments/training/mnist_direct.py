"""Experiment 05 T3: no-shadow direct projected SGD with exact codebook oracle."""

import csv
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.training.mnist_qat import MLP, data_split, evaluate
from experiments.training.tiny_oracle import codebooks


OUT = Path("experiments/results/procedural_training/mnist_direct")


def main(epochs=20, seeds=(11,17,23), learning_rate=.1,
         out=OUT):
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    images, labels, test_images, test_labels, train_ids, valid_ids, manifest = data_split()
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(images.reshape(60000,-1), dtype=torch.float32, device=device)/255
    y = torch.tensor(labels.astype("int64"), device=device)
    x_test = torch.tensor(test_images.reshape(10000,-1), dtype=torch.float32, device=device)/255
    y_test = torch.tensor(test_labels.astype("int64"), device=device)
    train_ids = torch.tensor(train_ids, device=device)
    valid_ids = torch.tensor(valid_ids, device=device)
    curves = []
    for seed in seeds:
        torch.manual_seed(seed)
        original = MLP().to(device)
        initial = {key:value.detach().clone() for key,value in original.state_dict().items()}
        scales = [float(layer.weight.detach().square().mean().sqrt()) for layer in original.layers]
        for method in ("scalar_direct", "procedural_direct"):
            book = codebooks()["scalar" if method=="scalar_direct" else "discovered"]
            model = MLP(book, scales).to(device)
            model.load_state_dict(initial, strict=False)
            with torch.no_grad():
                for layer, scale in zip(model.layers, scales):
                    layer.weight.copy_(model.nearest(layer.weight, scale)[0])
            optimizer = torch.optim.SGD([
                {"params": [layer.weight for layer in model.layers], "lr": learning_rate},
                {"params": [layer.bias for layer in model.layers], "lr": .1},
            ])
            start = time.perf_counter()
            for epoch in range(epochs+1):
                valid_ce, valid_accuracy = evaluate(model, x[valid_ids], y[valid_ids])
                row = dict(seed=seed, method=method, epoch=epoch,
                           validation_ce=valid_ce, validation_accuracy=valid_accuracy,
                           train_seconds=time.perf_counter()-start)
                if epoch==epochs:
                    test_ce, test_accuracy = evaluate(model, x_test, y_test)
                    row.update(test_ce=test_ce, test_accuracy=test_accuracy)
                curves.append(row)
                if epoch==epochs:
                    break
                model.train()
                changed = total = 0
                order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(seed*1000+epoch)).to(device)
                for chosen in order.split(256):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x[train_ids[chosen]])
                    loss = F.cross_entropy(logits, y[train_ids[chosen]])
                    loss.backward()
                    before = []
                    with torch.no_grad():
                        for layer, scale in zip(model.layers, scales):
                            before.append(model.nearest(layer.weight, scale)[1])
                    optimizer.step()
                    with torch.no_grad():
                        for layer, scale, old in zip(model.layers, scales, before):
                            decoded, indices = model.nearest(layer.weight, scale)
                            changed += int((indices!=old).sum())
                            total += indices.numel()
                            layer.weight.copy_(decoded)
                curves[-1]["changed_fraction_next_epoch"] = changed/total
                print(seed, method, epoch+1, valid_accuracy, changed/total, flush=True)
    with (out/"curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in curves for key in row}))
        writer.writeheader(); writer.writerows(curves)
    (out/"config.json").write_text(json.dumps(dict(epochs=epochs, seeds=seeds, batch_size=256,
       optimizer="projected SGD", weight_learning_rate=learning_rate,
       bias_learning_rate=.1, architecture=[784,256,128,10],
       scale="frozen layer RMS at paired dense initialization", device=str(device)), indent=2)+"\n")


if __name__ == "__main__":
    main()
