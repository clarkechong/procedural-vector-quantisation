"""Official E-means on real binary block masks, followed by matched balancing."""
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from grouping_screen import SOURCE, balanced_from_centroids, block_masks, cost, lloyd_balanced, original
from reference_emeans import load_official


def main():
    torch.set_num_threads(8)
    Emeans = load_official(Path('/tmp/faiss-emeans'))
    out = Path('experiments/results/experiments-22-26/emeans_masks.csv')
    rows = []
    for doc in (67, 68, 69, 72, 73):
        x = np.load(SOURCE / 'data/activations' / f'doc_{doc:04d}.npy')[:128]
        a = block_masks(x, 16)
        n, d = a.shape
        g, k = 8, n//8
        rng = np.random.default_rng(22+doc)
        init = a[rng.choice(n,k,replace=False)].astype('float32')
        started = time.perf_counter()
        e = Emeans(d,k,nepochs=3,seed=22+doc)
        c,_ = e.train(torch.from_numpy(a.astype('float32')),torch.from_numpy(init))
        elapsed = time.perf_counter()-started
        nearest = ((a[:,None]-c.numpy()[None])**2).sum(2).argmin(1)
        balanced = balanced_from_centroids(a.astype('float32'), c.numpy(),g)
        lloyd = lloyd_balanced(a,g,np.random.default_rng(22+doc))
        rows.append(dict(doc=doc,n=n,g=g,b=16,original=cost(a,original(n,g)),
                         official_balanced=cost(a,balanced),lloyd_balanced=cost(a,lloyd),
                         official_pre_balance_max_cluster=int(np.bincount(nearest,minlength=k).max()),
                         official_pre_balance_empty=int((np.bincount(nearest,minlength=k)==0).sum()),
                         official_fit_s=elapsed,official_batch_size=e.recipe['batch_size']))
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows))


if __name__=='__main__':main()
