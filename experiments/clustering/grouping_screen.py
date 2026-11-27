"""Bounded, exact-cost ProSparse mask grouping screen on historical documents.

All groups have exact capacity; each window lies within one document. This is an
offline upper bound, since the mask is observed after the gate/up projections.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

SOURCE = Path('/root/work/sparse-2-4')


def block_masks(x, b):
    n, d = x.shape
    padded = np.pad(x != 0, ((0, 0), (0, (-d) % b)))
    return padded.reshape(n, -1, b).any(axis=2)


def cost(a, groups):
    return sum(int(a[g].any(axis=0).sum()) for g in groups)


def original(n, g):
    assert n % g == 0
    return np.arange(n).reshape(-1, g).copy()


def greedy(a, g, rng):
    n = len(a)
    remaining = list(rng.permutation(n))
    groups = []
    while remaining:
        group = [remaining.pop(0)]
        union = a[group[0]].copy()
        while len(group) < g:
            incr = np.count_nonzero(a[remaining] & ~union, axis=1)
            j = int(np.argmin(incr))
            chosen = remaining.pop(j)
            group.append(chosen)
            union |= a[chosen]
        groups.append(group)
    return np.asarray(groups, dtype=np.int32)


def hamming_nn(a, g, rng):
    n = len(a)
    remaining = list(rng.permutation(n))
    groups = []
    while remaining:
        group = [remaining.pop(0)]
        while len(group) < g:
            distance = np.count_nonzero(a[remaining] != a[group[-1]], axis=1)
            group.append(remaining.pop(int(np.argmin(distance))))
        groups.append(group)
    return np.asarray(groups, dtype=np.int32)


def lloyd_balanced(a, g, rng, iters=5):
    """Lloyd on binary points with fractional centroids; greedy exact-capacity assignment."""
    n, _ = a.shape
    k = n // g
    x = a.astype(np.float32)
    centroids = x[rng.choice(n, k, replace=False)].copy()
    for _ in range(iters):
        groups = balanced_from_centroids(x, centroids, g)
        centroids = np.asarray([x[row].mean(axis=0) for row in groups])
    return np.asarray(groups, dtype=np.int32)


def balanced_from_centroids(x, centroids, g):
    """Exact-capacity greedy repair shared by Lloyd and E-means candidates."""
    n, k = len(x), len(centroids)
    assert n == k * g
    dist = ((x[:, None] - centroids[None]) ** 2).sum(axis=2)
    preference = np.argsort(dist, axis=1, kind='stable')
    margins = np.partition(dist, 1, axis=1)[:, 1] - np.min(dist, axis=1)
    order = np.argsort(-margins, kind='stable')
    groups = [[] for _ in range(k)]
    for i in order:
        j = next(j for j in preference[i] if len(groups[j]) < g)
        groups[j].append(int(i))
    return np.asarray(groups, dtype=np.int32)


def refine(a, groups, max_passes=2):
    """First-improvement exact legal swaps. Never accepts an increasing tile cost."""
    groups = groups.copy()
    k, g = groups.shape
    for _ in range(max_passes):
        changed = False
        for j in range(k):
            for h in range(j + 1, k):
                before = int(a[groups[j]].any(0).sum() + a[groups[h]].any(0).sum())
                best = (0, -1, -1)
                for u in range(g):
                    for v in range(g):
                        gj = groups[j].copy(); gh = groups[h].copy()
                        gj[u], gh[v] = gh[v], gj[u]
                        delta = int(a[gj].any(0).sum() + a[gh].any(0).sum()) - before
                        if delta < best[0]:
                            best = (delta, u, v)
                if best[0] < 0:
                    _, u, v = best
                    groups[j, u], groups[h, v] = groups[h, v], groups[j, u]
                    changed = True
        if not changed:
            break
    return groups


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []; captures = []
    for doc in args.docs:
        path = SOURCE / 'data/activations' / f'doc_{doc:04d}.npy'
        x = np.load(path, mmap_mode='r')
        meta = json.loads((SOURCE / 'data/captures' / f'doc_{doc:04d}.json').read_text())
        captures.append(dict(doc=doc, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             tokens=len(x), metadata=meta))
        for offset in range(0, len(x) - args.window + 1, args.window):
            window = np.asarray(x[offset:offset + args.window])
            for b in args.blocks:
                a = block_masks(window, b)
                for g in args.groups:
                    if len(a) % g: continue
                    rng = np.random.default_rng(args.seed + doc * 10000 + offset * 100 + b * 10 + g)
                    methods = {'original': original(len(a), g),
                               'random': rng.permutation(len(a)).reshape(-1, g),
                               'hamming_nn': hamming_nn(a, g, rng),
                               'greedy': greedy(a, g, rng),
                               'lloyd_balanced': lloyd_balanced(a, g, rng)}
                    best = min(methods, key=lambda name: cost(a, methods[name]))
                    methods['best_refined'] = refine(a, methods[best])
                    p = int((window != 0).sum())
                    channel = window != 0
                    for name, groups in methods.items():
                        assert np.array_equal(np.sort(groups.ravel()), np.arange(len(a)))
                        f_tile = cost(a, groups)
                        f_channel = cost(channel, groups)
                        rows.append(dict(doc=doc, offset=offset, n=len(a), g=g, b=b, method=name,
                                         p=p, density=p / window.size, f_channel=f_channel,
                                         f_tile=f_tile, issued_slots=f_tile*g*b,
                                         useful_fraction=p/(f_tile*g*b) if f_tile else 1,
                                         block_lower_bound=int(np.ceil(a.sum(0)/g).sum()),
                                         channel_lower_bound=int(np.ceil(channel.sum(0)/g).sum())))
    with (out/'grouping_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'captures_manifest.json').write_text(json.dumps(captures, indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows), windows=len({(r['doc'], r['offset']) for r in rows}), output=str(out))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--docs', nargs='+', type=int, default=[64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75])
    p.add_argument('--window', type=int, default=128)
    p.add_argument('--groups', nargs='+', type=int, default=[8, 16])
    p.add_argument('--blocks', nargs='+', type=int, default=[16, 32])
    p.add_argument('--seed', type=int, default=22)
    p.add_argument('--output', default='results/e22-26-emeans-prosparse')
    run(p.parse_args())
