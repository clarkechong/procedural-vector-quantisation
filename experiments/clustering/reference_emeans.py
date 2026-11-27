"""Run pinned upstream PyTorch E-means CPU component and matched Lloyd control.

Source is fetched separately into --source; no ROCm environment changes needed.
This synthetic test is an adaptation, not a reproduction of the paper benchmark.
"""
import argparse
import importlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


def load_official(path):
    package = types.ModuleType('emeans_official')
    package.__path__ = [str(path / 'contrib/e_means')]
    sys.modules[package.__name__] = package
    return importlib.import_module('emeans_official.e_means_torch').Emeans


def loss(x, c):
    d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(2)
    return float(d.min(1).mean()), d.argmin(1)


def lloyd(x, init, epochs):
    c = init.copy()
    trace = []
    start = time.perf_counter()
    for epoch in range(epochs):
        train_loss, labels = loss(x, c)
        for j in range(len(c)):
            if np.any(labels == j): c[j] = x[labels == j].mean(0)
        trace.append(dict(epoch=epoch + 1, train_loss=train_loss,
                          elapsed_s=time.perf_counter() - start))
    return c, trace


def main(a):
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    centers = rng.normal(size=(a.k, a.d)).astype('float32') * 3
    weights = np.arange(1, a.k + 1, dtype=np.float64) ** -1
    weights /= weights.sum()
    membership = rng.choice(a.k, a.n + a.test_n, p=weights)
    x = centers[membership] + rng.normal(scale=0.7, size=(a.n+a.test_n, a.d)).astype('float32')
    train, test = x[:a.n], x[a.n:]
    init = train[rng.choice(len(train), a.k, replace=False)].copy()
    started = time.perf_counter()
    e = load_official(Path(a.source))(a.d, a.k, nepochs=a.epochs, seed=a.seed)
    c, labels = e.train(torch.from_numpy(train), torch.from_numpy(init))
    elapsed = time.perf_counter() - started
    c = c.numpy()
    distances, assigned = e.assign(torch.from_numpy(test))
    exact_loss, exact_labels = loss(test.astype('float64'), c.astype('float64'))
    assert np.array_equal(assigned.numpy(), exact_labels)
    assert np.allclose(distances.numpy().mean(), exact_loss, rtol=1e-4, atol=1e-4)
    lloyd_c, lloyd_trace = lloyd(train, init, a.epochs)
    result = dict(source=a.source, seed=a.seed, n=a.n, test_n=a.test_n, d=a.d,k=a.k,
                  epochs=a.epochs, threads=a.threads, official_recipe=e.recipe,
                  official_cpu=dict(elapsed_s=elapsed, heldout_loss=exact_loss,
                                    exact_assignment_agreement=1.0,
                                    unused_clusters=int((np.bincount(labels.numpy(),minlength=a.k)==0).sum())),
                  lloyd_cpu=dict(heldout_loss=loss(test,lloyd_c)[0], trace=lloyd_trace))
    Path(a.output).write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps(dict(official=result['official_cpu'], lloyd_heldout=result['lloyd_cpu']['heldout_loss'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/tmp/faiss-emeans')
    p.add_argument('--output', default='experiments/results/experiments-22-26/reference_emeans.json')
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--test-n', type=int, default=1024)
    p.add_argument('--d', type=int, default=32)
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=22)
    main(p.parse_args())
