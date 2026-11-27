"""Experiment 07 pilot: learned target-only inverse versus exact and search baselines."""

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.training.tiny_oracle import codebooks


OUT = Path("results/e04-07-reference-optimization/learned-optimizer/inverse-pilot")


def samples(book, count, seed):
    rng = np.random.default_rng(seed)
    current = rng.integers(0, 256, size=count)
    # Mix local proximal targets and diffuse targets so trivial identity is insufficient.
    noise = rng.normal(size=(count, 4)) * rng.choice([0.2, 0.6, 1.2], size=(count, 1))
    target = book[current] + noise
    return target.astype(np.float32), current


def exact_cost(target, book):
    return np.sum((target[:, None, :] - book[None, :, :])**2, axis=2)


def train_and_evaluate(out=OUT, device=None):
    torch.set_num_threads(4)
    torch.manual_seed(20261127)
    out.mkdir(parents=True, exist_ok=True)
    book = codebooks()["discovered"].astype(np.float32)
    train_x, _ = samples(book, 50000, 17)
    valid_x, _ = samples(book, 5000, 23)
    test_x, current = samples(book, 10000, 31)
    train_y = exact_cost(train_x, book).argmin(axis=1)
    valid_y = exact_cost(valid_x, book).argmin(axis=1)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    network = torch.nn.Sequential(torch.nn.Linear(4, 128), torch.nn.ReLU(),
                                  torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                  torch.nn.Linear(128, 256)).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    features = torch.from_numpy(train_x).to(device)
    labels = torch.from_numpy(train_y).long().to(device)
    validation = torch.from_numpy(valid_x).to(device)
    valid_labels = torch.from_numpy(valid_y).long().to(device)
    history = []
    start = time.perf_counter()
    best = float("inf")
    best_state = None
    for epoch in range(12):
        network.train()
        order = torch.randperm(len(features), device=device)
        for batch in order.split(512):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(network(features[batch]), labels[batch])
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            valid_ce = float(torch.nn.functional.cross_entropy(network(validation), valid_labels))
        history.append(dict(epoch=epoch+1, validation_ce=valid_ce))
        if valid_ce < best:
            best = valid_ce
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
    training_seconds = time.perf_counter()-start
    network.load_state_dict(best_state)
    torch.save(best_state, out/"model.pt")
    network.eval()
    inference_start = time.perf_counter()
    with torch.no_grad():
        logits = torch.cat([network(batch.to(device)).cpu() for batch in torch.from_numpy(test_x).split(512)]).numpy()
    inference_seconds = time.perf_counter()-inference_start
    cost = exact_cost(test_x, book)
    exact = cost.min(axis=1)
    rng = np.random.default_rng(91)
    proposals = rng.integers(0, 256, size=(len(test_x), 16))
    rows = np.arange(len(test_x))[:, None]
    metrics = []
    for k in (1, 4, 16):
        top = np.argpartition(-logits, k-1, axis=1)[:, :k]
        candidates = np.column_stack((current, top))
        scores = cost[rows, candidates]
        predicted = scores.min(axis=1)
        metrics.append(dict(method=f"learned_top{k}+current", candidates=k+1,
                            mean_regret=float(np.mean(predicted-exact)),
                            oracle_hit=float(np.mean(np.isclose(predicted, exact, atol=1e-6)))))
    random_best = cost[rows, np.column_stack((current, proposals))].min(axis=1)
    metrics.append(dict(method="random16+current", candidates=17,
                        mean_regret=float(np.mean(random_best-exact)),
                        oracle_hit=float(np.mean(np.isclose(random_best, exact, atol=1e-6)))))
    flips = current[:, None] ^ (1 << np.arange(8))[None, :]
    flip_best = cost[rows, np.column_stack((current, flips))].min(axis=1)
    metrics.append(dict(method="one_bit+current", candidates=9,
                        mean_regret=float(np.mean(flip_best-exact)),
                        oracle_hit=float(np.mean(np.isclose(flip_best, exact, atol=1e-6)))))
    metrics.append(dict(method="exact_nearest", candidates=256,
                        mean_regret=0., oracle_hit=1.))
    for name, data in (("metrics.csv", metrics), ("history.csv", history)):
        with (out/name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    (out/"config.json").write_text(json.dumps(dict(train=50000, validation=5000, test=10000,
       generator="discovered exact 8-bit/4-weight", target="codeword plus Gaussian noise, sigma 0.2/0.6/1.2",
       seeds=dict(torch=20261127, train=17, validation=23, test=31, random=91), epochs=12,
       batch_size=512, device=device, training_seconds=training_seconds,
       inference_seconds=inference_seconds), indent=2)+"\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_and_evaluate()
