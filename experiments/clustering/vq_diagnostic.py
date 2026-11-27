"""Bounded CPU VQ diagnostic on disjoint ProSparse down-projection rows."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.spatial.distance import cdist

from reference_emeans import load_official

WEIGHT = Path('/root/work/sparse-2-4/data/weights/original.safetensors')


def assign(x, c):
    d = cdist(x, c, 'sqeuclidean')
    labels = d.argmin(1)
    return labels, float(d[np.arange(len(x)), labels].mean())


def lloyd(x, initial, epochs):
    c = initial.copy()
    trace=[]; start=time.perf_counter()
    for epoch in range(epochs):
        labels, _ = assign(x,c)
        for j in range(len(c)):
            if np.any(labels==j): c[j]=x[labels==j].mean(0)
        trace.append(dict(epoch=epoch+1,seconds=time.perf_counter()-start))
    return c,trace


def main():
    torch.set_num_threads(8)
    with safe_open(WEIGHT,framework='pt') as f:
        w=f.get_tensor('down_proj').numpy().astype('float32')
    rng=np.random.default_rng(26)
    d=4;k=256
    scale=float(np.sqrt(np.mean(w[:2048]**2)))
    train=w[:2048].reshape(-1,d)
    test=w[2048:].reshape(-1,d)
    train=train[rng.choice(len(train),16384,replace=False)]/scale
    test=test[rng.choice(len(test),4096,replace=False)]/scale
    initial=train[rng.choice(len(train),k,replace=False)].copy()
    e=load_official(Path('/tmp/faiss-emeans'))(d,k,nepochs=3,seed=26)
    start=time.perf_counter();ec,_=e.train(torch.from_numpy(train),torch.from_numpy(initial));et=time.perf_counter()-start
    lc,lt=lloyd(train,initial,3)
    result={'source':'ProSparse layer15 down_proj FP16, distinct output-row halves',
            'weight_sha256':hashlib.sha256(WEIGHT.read_bytes()).hexdigest(),
            'train_rows':'0:2048','test_rows':'2048:4096','train_tiles_sampled':len(train),
            'test_tiles_sampled':len(test),'d':d,'k':k,'scale':scale,
            'nominal_index_bpw':2.0,'table_bits_fp16':k*d*16,
            'methods':{}}
    for name,c,seconds in [('official_emeans',ec.numpy(),et),('lloyd',lc,lt[-1]['seconds'])]:
        labels,mse=assign(test,c)
        quant=c.astype('float16').astype('float32')
        _,quant_mse=assign(test,quant)
        result['methods'][name]=dict(seconds=seconds,test_nmse=mse/float((test**2).mean()*d),
                                     test_nmse_fp16_table=quant_mse/float((test**2).mean()*d),
                                     occupied_test_codes=int(np.unique(labels).size))
    out=Path('experiments/results/experiments-22-26/vq_diagnostic.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['methods']))


if __name__=='__main__':main()
