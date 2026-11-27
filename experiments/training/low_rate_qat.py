"""Paired low-rate QAT pilot with actual payload accounting.

Run with host GPU access: python -m experiments.training.low_rate_qat.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.generator_algebra.dictionary import hash32
from experiments.reference.qtip.codebooks import one_mad
from experiments.synthesis.trellis import signed_byte
from experiments.training.mnist_qat import MLP, data_split, evaluate


WEIGHTS = 234752
BIASES = 394


def make_book(family, width):
    seed = np.arange(256, dtype=np.uint32)[:, None]
    coordinate = np.arange(width, dtype=np.uint32)[None]
    if family == "mul120":
        # The first four coordinates exactly reproduce the earlier 8/4 book.
        word = (seed >> (coordinate % 4)) ^ ((coordinate // 4) * np.uint32(73))
        values = signed_byte(120 * word).astype(np.float32)
    elif family == "qtip_1mad":
        state = (seed << np.uint32(8)) ^ (coordinate * np.uint32(0x9e37))
        values = one_mad(state).astype(np.float32)
    elif family == "hash32":
        word = hash32(seed ^ (coordinate * np.uint32(0x9e3779b9)))
        values = signed_byte(word).astype(np.float32)
    elif family == "random_vq":
        values = np.random.default_rng(9127 + width).standard_normal((256, width)).astype(np.float32)
    else:
        raise ValueError(family)
    values -= values.mean()
    values /= np.sqrt(np.mean(values.astype(np.float64) ** 2))
    return values


class QuantMLP(MLP):
    def __init__(self, method, scales, book=None, width=4):
        super().__init__()
        self.method = method
        self.scales = scales
        self.width = width
        if book is not None:
            if isinstance(book, tuple):
                self.register_buffer("codebook", torch.as_tensor(book[0], dtype=torch.float32))
                self.register_buffer("second_book", torch.as_tensor(book[1], dtype=torch.float32))
            else:
                self.register_buffer("codebook", torch.as_tensor(book, dtype=torch.float32))
                self.register_buffer("second_book", None)
        else:
            self.register_buffer("codebook", None)
            self.register_buffer("second_book", None)

    def quantize(self, weight, scale):
        if self.method == "dense":
            return weight
        if self.method.startswith("scalar") or self.method in ("ternary", "binary"):
            levels = {"scalar4": list(range(-15, 16, 2)),
                      "scalar2": [-3, -1, 1, 3],
                      "ternary": [-1, 0, 1], "binary": [-1, 1]}[self.method]
            levels = torch.tensor(levels, dtype=weight.dtype, device=weight.device)
            levels = levels / levels.square().mean().sqrt()
            indices = (weight[..., None] / scale - levels).abs().argmin(-1)
            return levels[indices] * scale
        def project(values, width, book):
            tiles = values.reshape(-1, width) / scale
            scores = book.square().sum(-1)[None, :] - 2 * tiles @ book.T
            return book[scores.argmin(-1)].reshape_as(values) * scale
        if self.width == 0:
            flat = weight.flatten()
            half = flat.numel() // 2
            return torch.cat((project(flat[:half], 4, self.codebook),
                              project(flat[half:], 8, self.second_book))).reshape_as(weight)
        return project(weight, self.width, self.codebook)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            weight = layer.weight
            if self.method != "dense":
                with torch.no_grad():
                    quantized = self.quantize(weight, self.scales[i])
                weight = weight + (quantized - weight).detach()
            x = F.linear(x, weight, layer.bias)
            if i < 2:
                x = F.relu(x)
        return x


class LowRankMLP(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for input_size, output_size in ((784, 256), (256, 128), (128, 10)):
            self.layers.append(torch.nn.Sequential(torch.nn.Linear(input_size, rank, bias=False),
                                                   torch.nn.Linear(rank, output_size)))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < 2:
                x = F.relu(x)
        return x


def storage(method, width):
    if method == "dense":
        return 32 * (WEIGHTS + BIASES)
    if method.startswith("lowrank"):
        rank = int(method.removeprefix("lowrank"))
        return 32 * (rank * (784+256+256+128+128+10) + BIASES)
    if method.startswith("scalar"):
        payload = {"scalar4": 4, "scalar2": 2}[method] * WEIGHTS
    elif method == "ternary":
        # Base-3 packed stream, rounded up per matrix.
        payload = sum(int(np.ceil(n * np.log2(3))) for n in (784*256, 256*128, 128*10))
    elif method == "binary":
        payload = WEIGHTS
    else:
        payload = int(WEIGHTS * 1.5) if width == 0 else (WEIGHTS // width) * 8
        if method.startswith("random_vq"):
            payload += 256 * (12 if width == 0 else width) * 32  # Explicit shared FP32 codebook.
    return int(payload + 32 * (BIASES + 3))


def run(args):
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("GPU required")
    configs = [("dense", None, None), ("scalar4", None, None),
               ("scalar2", None, None), ("ternary", None, None),
               ("binary", None, None)]
    configs += [(f"lowrank{rank}", None, None) for rank in (1, 2, 4, 8)]
    for width in (2, 4, 8, 16, 32):
        for family in ("mul120", "qtip_1mad", "hash32", "random_vq"):
            configs.append((f"{family}_{8/width:g}bpw", family, width))
    for family in ("mul120", "qtip_1mad", "hash32", "random_vq"):
        configs.append((f"{family}_1.5bpw", family, 0))
    if args.methods:
        configs = [c for c in configs if c[0] in args.methods]
    rows = []
    manifest = None
    for dataset in args.datasets:
        root = Path("data") / dataset
        images, labels, test_images, test_labels, train_ids, valid_ids, manifest = data_split(root)
        x = torch.tensor(images.reshape(60000, -1), dtype=torch.float32, device=device) / 255
        y = torch.tensor(labels.astype(np.int64), device=device)
        x_test = torch.tensor(test_images.reshape(10000, -1), dtype=torch.float32, device=device) / 255
        y_test = torch.tensor(test_labels.astype(np.int64), device=device)
        train_ids = torch.tensor(train_ids, device=device)
        valid_ids = torch.tensor(valid_ids, device=device)
        for seed in args.seeds:
            torch.manual_seed(seed)
            initial_model = MLP().to(device)
            initial = {k: v.detach().clone() for k, v in initial_model.state_dict().items()}
            scales = [float(layer.weight.detach().square().mean().sqrt()) for layer in initial_model.layers]
            for name, family, width in configs:
                book = ((make_book(family, 4), make_book(family, 8)) if width == 0 else
                        make_book(family, width)) if family else None
                if name.startswith("lowrank"):
                    torch.manual_seed(seed)
                    model = LowRankMLP(int(name.removeprefix("lowrank"))).to(device)
                else:
                    model = QuantMLP(name if family is None else "procedural", scales, book,
                                     4 if width is None else width).to(device)
                    model.load_state_dict(initial, strict=False)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                start = time.perf_counter()
                for epoch in range(args.epochs):
                    model.train()
                    order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(seed*1000+epoch)).to(device)
                    for chosen in order.split(256):
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(x[train_ids[chosen]]), y[train_ids[chosen]])
                        loss.backward()
                        optimizer.step()
                val_ce, val_acc = evaluate(model, x[valid_ids], y[valid_ids])
                test_ce, test_acc = evaluate(model, x_test, y_test)
                bits = storage(name, width)
                row = dict(dataset=dataset, seed=seed, method=name, family=family or "",
                           width=width or 1, epochs=args.epochs, validation_ce=val_ce,
                           validation_accuracy=val_acc, test_ce=test_ce,
                           test_accuracy=test_acc, stored_bits=bits,
                           stored_bpw=bits / WEIGHTS, elapsed_s=time.perf_counter()-start)
                rows.append(row)
                with (out / "metrics.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                    writer.writeheader(); writer.writerows(rows)
                print(dataset, seed, name, test_acc, row["stored_bpw"], flush=True)
    (out / "config.json").write_text(json.dumps(dict(datasets=args.datasets, seeds=args.seeds,
        epochs=args.epochs, architecture=[784,256,128,10], optimizer="Adam lr 0.001",
        batch_size=256, scales="frozen paired initialization RMS", device=str(device),
        split=manifest, note="Test rows are previously exposed; exploratory, not blind confirmation"), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion-mnist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 17, 23])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--out", default="experiments/results/procedural_training/low_rate_qat")
    parser.add_argument("--require-gpu", action="store_true")
    run(parser.parse_args())
