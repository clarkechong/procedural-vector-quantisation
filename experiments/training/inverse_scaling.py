"""Bitwise inverse scaling pilot against exhaustive nearest-codeword oracles."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.generator_algebra.dictionary import hash32


WIDTH = 16


def codebook(bits, family):
    seed = np.arange(1 << bits, dtype=np.uint32)[:, None]
    coordinate = np.arange(WIDTH, dtype=np.uint32)[None]
    if family == "bit_extract":
        values = (((seed >> (coordinate % bits)) & 1).astype(np.float32) * 2 - 1)
    elif family == "linear_binary":
        first = coordinate % bits
        mask = (np.uint32(1) << first) ^ np.where(first + 1 < bits,
                    np.uint32(1) << ((first + 1) % bits), np.uint32(0))
        values = (np.bitwise_count(seed & mask) % 2).astype(np.float32) * 2 - 1
    elif family == "hash32":
        word = hash32(seed ^ (coordinate * np.uint32(0x9e3779b9)))
        values = ((word & 255).astype(np.float32) - 127.5) / 73.9
    elif family == "random_explicit":
        values = np.random.default_rng(991 + bits).standard_normal((1 << bits, WIDTH)).astype(np.float32)
    else:
        raise ValueError(family)
    return np.broadcast_to(values, (1 << bits, WIDTH)).copy()


def exact(book, targets, chunk=65536):
    """Exhaustive nearest search with bounded score matrix and exact float32 costs."""
    best = torch.full((len(targets),), float("inf"), device=targets.device)
    index = torch.zeros(len(targets), dtype=torch.long, device=targets.device)
    for start in range(0, len(book), chunk):
        section = book[start:start+chunk]
        costs = (section.square().sum(-1)[:, None] - 2 * section @ targets.T)
        value, local = costs.min(dim=0)
        take = value < best
        best = torch.where(take, value, best)
        index = torch.where(take, local + start, index)
    true_cost = (book[index] - targets).square().sum(-1)
    return index, true_cost


class Inverse(torch.nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(WIDTH, 256), torch.nn.ReLU(),
                                       torch.nn.Linear(256, 256), torch.nn.ReLU(),
                                       torch.nn.Linear(256, bits))

    def forward(self, x):
        return self.net(x)


def run(args):
    torch.set_num_threads(4)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("GPU required")
    rows = []
    for bits in args.widths:
        for family in args.families:
            if device.type=="cuda": torch.cuda.synchronize()
            begin = time.perf_counter()
            book = torch.tensor(codebook(bits, family), device=device)
            rng = np.random.default_rng(81620 + bits)
            train_seed = torch.tensor(rng.integers(0, len(book), args.train_count), device=device)
            test_seed = torch.tensor(rng.integers(0, len(book), args.test_count), device=device)
            generator = torch.Generator(device=device).manual_seed(991 + bits)
            train_x = book[train_seed] + args.noise * torch.randn((args.train_count, WIDTH), device=device, generator=generator)
            test_x = book[test_seed] + args.noise * torch.randn((args.test_count, WIDTH), device=device, generator=generator)
            torch.manual_seed(20261129)
            model = Inverse(bits).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            powers = 1 << torch.arange(bits, device=device)
            train_bits = ((train_seed[:, None] & powers) != 0).float()
            for epoch in range(args.epochs):
                order = torch.randperm(args.train_count, device=device)
                for chosen in order.split(512):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.binary_cross_entropy_with_logits(model(train_x[chosen]), train_bits[chosen])
                    loss.backward(); optimizer.step()
            if device.type=="cuda": torch.cuda.synchronize()
            train_s = time.perf_counter() - begin
            with torch.no_grad():
                logits = model(test_x)
                predicted = ((logits > 0).long() * powers).sum(-1)
                predicted_cost = (book[predicted] - test_x).square().sum(-1)
                # Learned proposal and each one-bit neighbor: B+1 candidate evaluations.
                neighbors = predicted[:, None] ^ torch.cat((torch.zeros(1,device=device,dtype=torch.long), powers))
                candidate_cost = (book[neighbors] - test_x[:, None]).square().sum(-1)
                best_cost, best_column = candidate_cost.min(-1)
                best_seed = neighbors.gather(1, best_column[:, None])[:, 0]
                oracle_count = min(args.oracle_count, args.test_count)
                if device.type=="cuda": torch.cuda.synchronize()
                oracle_start = time.perf_counter()
                oracle_seed, oracle_cost = exact(book, test_x[:oracle_count])
                if device.type=="cuda": torch.cuda.synchronize()
                oracle_s = time.perf_counter() - oracle_start
                regret = best_cost[:oracle_count] - oracle_cost
                row = dict(bits=bits, family=family, train_count=args.train_count,
                    test_count=args.test_count, oracle_count=oracle_count,
                    train_s=train_s, oracle_s=oracle_s,
                    exact_seed_rate=float((best_seed[:oracle_count] == oracle_seed).float().mean()),
                    planted_seed_rate=float((predicted == test_seed).float().mean()),
                    mean_predicted_mse=float(predicted_cost.mean()/WIDTH),
                    mean_candidate_mse=float(best_cost.mean()/WIDTH),
                    mean_oracle_mse=float(oracle_cost.mean()/WIDTH),
                    mean_regret=float(regret.mean()/WIDTH),
                    candidate_evaluations=bits+1,
                    exhaustive_evaluations=1<<bits,
                    explicit_codebook_bits=(1<<bits)*WIDTH*32 if family=="random_explicit" else 0)
                rows.append(row)
                with (out/"metrics.csv").open("w",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
                print(bits, family, row["exact_seed_rate"], row["mean_regret"], flush=True)
    (out/"config.json").write_text(json.dumps(dict(widths=args.widths, families=args.families,
        width=WIDTH, train_count=args.train_count, test_count=args.test_count,
        oracle_count=args.oracle_count, epochs=args.epochs, noise=args.noise,
        device=str(device), training_target="planted seed, not nearest-codeword oracle",
        input_distribution="sampled codeword plus IID Gaussian noise",
        candidate_rule="threshold bit predictor plus one-bit neighborhood",
        model="16-256-256-B ReLU", note="Synthetic scaling pilot, not task-gradient inversion"),indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--widths",nargs="+",type=int,default=[12,16,20])
    parser.add_argument("--families",nargs="+",default=["bit_extract","linear_binary","hash32","random_explicit"])
    parser.add_argument("--train-count",type=int,default=20000)
    parser.add_argument("--test-count",type=int,default=2000)
    parser.add_argument("--oracle-count",type=int,default=64)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--noise",type=float,default=0.25)
    parser.add_argument("--out",default="experiments/results/learned_optimizer/inverse_scaling")
    parser.add_argument("--require-gpu",action="store_true")
    run(parser.parse_args())
