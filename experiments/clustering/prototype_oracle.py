"""Cheap offline prototype-mask oracle on real ProSparse intermediate activations.

Requires GPU for full down-projection output checks. This is an oracle using
intermediate h and cannot be deployed as a router from FFN input x.
"""
import hashlib
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.cluster.vq import kmeans2

ROOT=Path('/root/work/sparse-2-4')


def top_blocks(h, b, q):
    energy=(h.reshape(len(h),-1,b).astype('float32')**2).sum(2)
    idx=np.argpartition(energy,-q,axis=1)[:,-q:]
    mask=np.zeros_like(energy,dtype=bool)
    np.put_along_axis(mask,idx,True,axis=1)
    return mask


def main(q):
    assert torch.cuda.is_available(), 'Run on host with GPU device nodes exposed'
    torch.set_num_threads(8)
    b=16;k=16;seed=25
    rng=np.random.default_rng(seed)
    train=[]
    for doc in range(32):
        h=np.load(ROOT/'data/activations'/f'doc_{doc:04d}.npy',mmap_mode='r')
        if len(h):train.append(np.asarray(h[rng.choice(len(h),min(32,len(h)),replace=False)]))
    train=np.concatenate(train)
    train_mask=top_blocks(train,b,q).astype('float32')
    start=time.perf_counter()
    _,labels=kmeans2(train_mask,k,iter=12,minit='points',seed=seed)
    prototypes=np.zeros((k,train_mask.shape[1]),bool)
    for j in range(k):
        scores=train_mask[labels==j].mean(0) if np.any(labels==j) else train_mask.mean(0)
        prototypes[j,np.argpartition(scores,-q)[-q:]]=True
    fit_s=time.perf_counter()-start
    docs=[65,67,68,69,70,72,73,75]
    test=[];counts=[]
    for doc in docs:
        h=np.load(ROOT/'data/activations'/f'doc_{doc:04d}.npy',mmap_mode='r')
        chunk=np.asarray(h[:min(8,len(h))]);test.append(chunk);counts.append(len(chunk))
    test=np.concatenate(test)
    device='cuda';w=load_file(ROOT/'data/weights/original.safetensors')['down_proj'].to(device)
    h=torch.from_numpy(test.copy()).to(device)
    with torch.inference_mode():
        y=torch.nn.functional.linear(h,w).float()
        den=y.square().sum(1).clamp_min(1e-20)
        errors=[]
        for p in prototypes:
            m=torch.from_numpy(np.repeat(p,b)).to(device)
            yp=torch.nn.functional.linear(h*m,w).float()
            errors.append((yp-y).square().sum(1)/den)
        oracle=torch.stack(errors).min(0).values
        per_token=torch.from_numpy(np.repeat(top_blocks(test,b,q),b,axis=1)).to(device)
        top_y=torch.nn.functional.linear(h*per_token,w).float()
        top_error=(top_y-y).square().sum(1)/den
        fixed=torch.from_numpy(np.repeat(prototypes[0],b)).to(device)
        fixed_y=torch.nn.functional.linear(h*fixed,w).float()
        fixed_error=(fixed_y-y).square().sum(1)/den
    result=dict(source='ProSparse layer15 historical captures',train_docs=list(range(32)),
                evaluation_docs=docs,evaluation_token_counts=counts,seed=seed,b=b,q=q,
                kept_block_fraction=q/prototypes.shape[1],prototypes=k,
                train_tokens=len(train),fit_s=fit_s,
                oracle_output_nmse_mean=float(oracle.mean()),
                topq_output_nmse_mean=float(top_error.mean()),
                fixed_output_nmse_mean=float(fixed_error.mean()),
                oracle_output_nmse_p95=float(torch.quantile(oracle,.95)),
                topq_output_nmse_p95=float(torch.quantile(top_error,.95)))
    out=Path(f'results/e22-26-emeans-prosparse/prototype-oracle-q{q}.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--q',type=int,default=172)
    main(p.parse_args().q)
