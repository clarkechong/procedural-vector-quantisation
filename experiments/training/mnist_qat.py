"""Matched MNIST dense, scalar QAT, and fixed procedural shadow-STE training."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.training.tiny_oracle import codebooks


OUT = Path("experiments/results/procedural_training/mnist_qat")


def read_idx(path):
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    zero, dtype, dimensions = raw[:2], raw[2], raw[3]
    if zero != b"\0\0" or dtype != 8:
        raise ValueError("invalid unsigned-byte IDX")
    shape = struct.unpack(">"+"I"*dimensions, raw[4:4+4*dimensions])
    return np.frombuffer(raw, dtype=np.uint8, offset=4+4*dimensions).reshape(shape).copy()


def data_split(root=Path("data/mnist")):
    images = read_idx(root/"train-images-idx3-ubyte.gz")
    labels = read_idx(root/"train-labels-idx1-ubyte.gz")
    test_images = read_idx(root/"t10k-images-idx3-ubyte.gz")
    test_labels = read_idx(root/"t10k-labels-idx1-ubyte.gz")
    if images.shape != (60000, 28, 28) or test_images.shape != (10000, 28, 28):
        raise ValueError("unexpected MNIST shape")
    rng = np.random.default_rng(20261127)
    validation = np.concatenate([rng.choice(np.where(labels==digit)[0], 500, replace=False)
                                  for digit in range(10)])
    training = np.setdiff1d(np.arange(60000), validation)
    manifest = dict(dataset=str(root), train_count=len(training), validation_count=len(validation), test_count=10000,
       train_sha256=hashlib.sha256(training.tobytes()).hexdigest(),
       validation_sha256=hashlib.sha256(validation.tobytes()).hexdigest(),
       split_seed=20261127, validation_per_class=500,
       files={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in root.glob("*.gz")})
    return images, labels, test_images, test_labels, training, validation, manifest


class MLP(torch.nn.Module):
    def __init__(self, book=None, scales=None):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(784,256), torch.nn.Linear(256,128),
                                           torch.nn.Linear(128,10)])
        if book is None:
            self.register_buffer("book", None)
        else:
            self.register_buffer("book", torch.tensor(book, dtype=torch.float32))
        self.scales = scales

    def nearest(self, matrix, scale):
        tiles = matrix.reshape(-1,4)/scale
        norms = (self.book**2).sum(dim=1)
        scores = norms[None] - 2*tiles @ self.book.T
        indices = scores.argmin(dim=1)
        return self.book[indices].reshape(matrix.shape)*scale, indices

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            weight = layer.weight
            if self.book is not None:
                with torch.no_grad():
                    quantized, _ = self.nearest(weight, self.scales[i])
                weight = weight + (quantized-weight).detach()
            x = F.linear(x, weight, layer.bias)
            if i < 2:
                x = F.relu(x)
        return x


def evaluate(model, x, y):
    model.eval()
    loss = correct = 0.
    with torch.no_grad():
        for indices in torch.arange(len(x), device=x.device).split(512):
            logits = model(x[indices])
            loss += float(F.cross_entropy(logits, y[indices], reduction="sum"))
            correct += int((logits.argmax(dim=1)==y[indices]).sum())
    return loss/len(x), correct/len(x)


def main(epochs=20, seeds=(11,17,23), dataset_root=Path("data/mnist"), out=OUT):
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    images, labels, test_images, test_labels, train_ids, valid_ids, manifest = data_split(dataset_root)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(images.reshape(60000,-1), dtype=torch.float32, device=device)/255
    y = torch.tensor(labels.astype(np.int64), device=device)
    x_test = torch.tensor(test_images.reshape(10000,-1), dtype=torch.float32, device=device)/255
    y_test = torch.tensor(test_labels.astype(np.int64), device=device)
    train_ids = torch.tensor(train_ids, device=device)
    valid_ids = torch.tensor(valid_ids, device=device)
    books = codebooks()
    curves = []
    exports = []
    for seed in seeds:
        torch.manual_seed(seed)
        original = MLP().to(device)
        initial = {key:value.detach().clone() for key,value in original.state_dict().items()}
        rms = [float(layer.weight.detach().square().mean().sqrt()) for layer in original.layers]
        for method in ("dense", "scalar_qat", "procedural_qat"):
            book = None if method=="dense" else books["scalar" if method=="scalar_qat" else "discovered"]
            model = MLP(book, rms if book is not None else None).to(device)
            model.load_state_dict(initial, strict=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            start = time.perf_counter()
            for epoch in range(epochs+1):
                valid_ce, valid_accuracy = evaluate(model, x[valid_ids], y[valid_ids])
                # Test is evaluated only once at the preregistered final epoch.
                row = dict(seed=seed, method=method, epoch=epoch, validation_ce=valid_ce,
                           validation_accuracy=valid_accuracy, train_seconds=time.perf_counter()-start)
                if epoch==epochs:
                    test_ce, test_accuracy = evaluate(model, x_test, y_test)
                    row.update(test_ce=test_ce, test_accuracy=test_accuracy)
                    if book is not None:
                        total_bits = 0
                        max_error = 0.
                        used = set()
                        with torch.no_grad():
                            for layer, scale in zip(model.layers, rms):
                                reconstructed, indices = model.nearest(layer.weight, scale)
                                decoded = model.book[indices].reshape(layer.weight.shape)*scale
                                max_error = max(max_error, float((reconstructed-decoded).abs().max()))
                                used.update(indices.unique().cpu().tolist())
                                total_bits += indices.numel()*8 + 32 + layer.bias.numel()*32
                        exports.append(dict(seed=seed, method=method, deployed_bits=total_bits,
                                deployed_bpw=total_bits/234752, codewords_used=len(used),
                                max_decoder_error=max_error, training_shadow_weights=234752,
                                training_optimizer="Adam, dense shadow and first/second moments"))
                curves.append(row)
                if epoch==epochs:
                    break
                model.train()
                order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(seed*1000+epoch)).to(device)
                for chosen in order.split(256):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x[train_ids[chosen]])
                    loss = F.cross_entropy(logits, y[train_ids[chosen]])
                    loss.backward()
                    optimizer.step()
                print(seed, method, epoch+1, valid_accuracy, flush=True)
    for filename, rows in (("curves.csv", curves), ("export.csv", exports)):
        with (out/filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    (out/"config.json").write_text(json.dumps(dict(epochs=epochs, seeds=seeds, batch_size=256,
       optimizer="Adam", learning_rate=1e-3, architecture=[784,256,128,10],
       scale="frozen layer RMS at paired dense initialization", device=str(device)), indent=2)+"\n")


if __name__ == "__main__":
    main()
