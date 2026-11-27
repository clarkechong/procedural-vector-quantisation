"""Sustained hard-forward, latent-seed QAT with exact discrete Jacobians."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.training.low_rate_qat import make_book, WEIGHTS, BIASES
from experiments.training.mnist_qat import MLP, data_split, evaluate


class HardSeed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, book, method):
        bits = (logits > 0).to(torch.long)
        powers = (1 << torch.arange(8, device=logits.device))
        seed = (bits * powers).sum(-1)
        ctx.save_for_backward(seed, book, logits)
        ctx.method = method
        return book[seed]

    @staticmethod
    def backward(ctx, grad_output):
        seed, book, logits = ctx.saved_tensors
        if ctx.method == "discrete_jacobian":
            columns = []
            for bit in range(8):
                off = seed & ~(1 << bit)
                difference = book[off | (1 << bit)] - book[off]
                columns.append((difference * grad_output).sum(-1))
            grad = torch.stack(columns, -1)
            derivative = (1 - logits.abs()).clamp(min=0.05)
            return grad * derivative, None, None
        p = logits.sigmoid()
        if ctx.method == "boolean_degree2":
            # Mobius coefficients of the exact map, truncated to degree two.
            zero = book[0]
            first = torch.stack([book[1 << j] - zero for j in range(8)])
            columns = []
            for j in range(8):
                derivative = first[j].expand_as(grad_output).clone()
                for k in range(8):
                    if k != j:
                        pair = book[(1 << j) | (1 << k)] - book[1 << j] - book[1 << k] + zero
                        derivative += p[:, k:k+1] * pair
                columns.append((derivative * grad_output).sum(-1))
            grad = torch.stack(columns, -1) * p * (1-p)
            return grad, None, None
        if ctx.method == "bernoulli_exact":
            states = torch.arange(256, device=book.device)
            bits = ((states[:, None] >> torch.arange(8, device=book.device)) & 1).to(p.dtype)
            columns = []
            # Chunking bounds temporary probability storage.
            for start in range(0, len(p), 4096):
                part = p[start:start+4096]
                prob = torch.prod(torch.where(bits[None].bool(), part[:, None], 1-part[:, None]), -1)
                target = grad_output[start:start+4096]
                local = []
                for j in range(8):
                    derivative = prob * (bits[None, :, j] - part[:, j:j+1])
                    local.append(((derivative @ book) * target).sum(-1))
                columns.append(torch.stack(local, -1))
            return torch.cat(columns), None, None
        raise ValueError(ctx.method)


class LatentMLP(torch.nn.Module):
    def __init__(self, original, family="mul120", width=4, method="discrete_jacobian"):
        super().__init__()
        if width not in (4, 8, 16, 32):
            raise ValueError("Tile width must divide every weight matrix")
        self.width = width
        self.register_buffer("book", torch.tensor(make_book(family, width), dtype=torch.float32,
                                                   device=original.layers[0].weight.device))
        self.scales = [float(layer.weight.detach().square().mean().sqrt()) for layer in original.layers]
        self.method = method
        self.logits = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        self.shapes = []
        for layer, scale in zip(original.layers, self.scales):
            weight = layer.weight.detach()
            targets = weight.reshape(-1, width) / scale
            scores = self.book.square().sum(-1)[None] - 2 * targets @ self.book.T
            seed = scores.argmin(-1)
            bits = ((seed[:, None] >> torch.arange(8, device=seed.device)) & 1).float()
            self.logits.append(torch.nn.Parameter((bits * 2 - 1) * 0.25))
            self.biases.append(torch.nn.Parameter(layer.bias.detach().clone()))
            self.shapes.append(tuple(weight.shape))

    def forward(self, x):
        for i in range(3):
            weight = HardSeed.apply(self.logits[i], self.book, self.method).reshape(self.shapes[i]) * self.scales[i]
            x = F.linear(x, weight, self.biases[i])
            if i < 2:
                x = F.relu(x)
        return x


def run(args):
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("GPU required")
    rows = []
    for dataset in args.datasets:
        root = Path("data") / dataset
        images, labels, test_images, test_labels, train_ids, valid_ids, manifest = data_split(root)
        x = torch.tensor(images.reshape(60000, -1), dtype=torch.float32, device=device) / 255
        y = torch.tensor(labels.astype(np.int64), device=device)
        x_test = torch.tensor(test_images.reshape(10000, -1), dtype=torch.float32, device=device) / 255
        y_test = torch.tensor(test_labels.astype(np.int64), device=device)
        subset_rng = np.random.default_rng(20261129)
        train_ids = torch.tensor(subset_rng.choice(train_ids, args.train_count, replace=False), device=device)
        valid_ids = torch.tensor(valid_ids, device=device)
        for seed in args.seeds:
            torch.manual_seed(seed)
            original = MLP().to(device)
            model = LatentMLP(original, args.family, width=args.width, method=args.method).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            start = time.perf_counter()
            previous = [p.detach().clone() for p in model.logits]
            for epoch in range(args.epochs + 1):
                valid_ce, valid_acc = evaluate(model, x[valid_ids], y[valid_ids])
                test_ce, test_acc = evaluate(model, x_test, y_test) if epoch == args.epochs else (np.nan, np.nan)
                changed = sum(int(((p > 0) != (q > 0)).sum()) for p, q in zip(model.logits, previous))
                row = dict(dataset=dataset, seed=seed, method=args.method, epoch=epoch,
                           validation_ce=valid_ce, validation_accuracy=valid_acc,
                           test_ce=test_ce, test_accuracy=test_acc,
                           bit_flips_since_previous=changed, elapsed_s=time.perf_counter()-start)
                rows.append(row)
                with (out / "metrics.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                    writer.writeheader(); writer.writerows(rows)
                print(dataset, seed, epoch, valid_acc, changed, flush=True)
                if epoch == args.epochs:
                    break
                previous = [p.detach().clone() for p in model.logits]
                model.train()
                order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(seed*1000+epoch)).to(device)
                for chosen in order.split(256):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(x[train_ids[chosen]]), y[train_ids[chosen]])
                    loss.backward()
                    optimizer.step()
    latent_parameters = WEIGHTS // args.width * 8 + BIASES
    (out / "config.json").write_text(json.dumps(dict(datasets=args.datasets, seeds=args.seeds,
        epochs=args.epochs, train_count=args.train_count, architecture=[784,256,128,10],
        optimizer=f"Adam lr={args.lr}", family=args.family, method=args.method, width=args.width,
        device=str(device), split=manifest,
        latent_continuous_parameters=latent_parameters, shadow_continuous_parameters=WEIGHTS+BIASES,
        optimizer_moment_bytes_latent=8*latent_parameters,
        optimizer_moment_bytes_shadow=8*(WEIGHTS+BIASES),
        deployed_bits=WEIGHTS // args.width * 8+32*(BIASES+3),
        note="Allocated training memory excludes generated weights, gradients, activations and kernel workspaces; test rows previously exposed"), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion-mnist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 17, 23])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-count", type=int, default=55000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--family", default="mul120")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--method", choices=["discrete_jacobian", "boolean_degree2", "bernoulli_exact"],
                        default="discrete_jacobian")
    parser.add_argument("--out", default="results/e08-12-low-rate-training/procedural-training/latent-qat")
    parser.add_argument("--require-gpu", action="store_true")
    run(parser.parse_args())
