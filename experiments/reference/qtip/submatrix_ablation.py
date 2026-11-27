"""Matched 16x256 SmolLM2 QTIP-map/RHT/LDLQ local functional ablation."""

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.linalg import hadamard
from tokenizers import Tokenizer

from experiments.evaluation.seedlm_model import load_model,token_ids
from experiments.reference.qtip.codebooks import one_mad
from experiments.reference.qtip.pipeline_components import block_ldl
from experiments.reference.qtip.tailbite import quantize,pack_path,unpack_path
from experiments.synthesis.trellis import signed_byte


def orthogonal(n,rng):
    signs=rng.choice([-1.,1.],size=n)
    return signs[:,None]*hadamard(n).astype(np.float64)/np.sqrt(n),signs


def quantize_blocks(matrix,H,mapping,scale,compensate):
    m,n=matrix.shape
    result=np.zeros_like(matrix)
    payload=[]
    if compensate:
        regularized=H+np.eye(n)*max(float(np.diag(H).mean())*0.01,1e-12)
        L,_=block_ldl(torch.tensor(regularized),16)
        L=L.numpy()
    else:
        L=None
    for start in reversed(range(0,n,16)):
        target=matrix[:,start:start+16].copy()
        if compensate and start+16<n:
            target+=(matrix[:,start+16:]-result[:,start+16:])@L[start+16:,start:start+16]
        decoded,path=quantize(target.reshape(-1),mapping*scale)
        packed=pack_path(path)
        np.testing.assert_array_equal(unpack_path(packed,256),path)
        payload.append(packed)
        result[:,start:start+16]=decoded.reshape(m,16)
    return result,b"".join(payload)


def run(out=Path("experiments/results/reference/qtip-submatrix-ablation")):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=Tokenizer.from_file("data/smollm2/tokenizer.json")
    tokens=token_ids(Path("data/smollm2"),"validation",tokenizer)[:1024]
    model=load_model(Path("data/smollm2-135m.safetensors"),Path("data/smollm2"),torch.float32).to(device)
    module=model.model.layers[15].mlp.down_proj
    weight=module.weight.detach().cpu().numpy().astype(np.float64)[:16,:256]
    captured=[]
    def hook(_,args):captured.append(args[0].detach().float().cpu().reshape(-1,args[0].shape[-1]).numpy()[:,:256])
    handle=module.register_forward_pre_hook(hook)
    with torch.no_grad():
        for block in tokens.split(128):model(block[None].to(device),use_cache=False)
    handle.remove();del model
    x=np.concatenate(captured).astype(np.float64)
    xcal,xval=x[:512],x[512:]
    H=xcal.T@xcal/len(xcal)
    teacher=xval@weight.T
    rng=np.random.default_rng(20261127)
    left,left_signs=orthogonal(16,rng);right,right_signs=orthogonal(256,rng)
    if np.max(np.abs(left.T@(left@weight@right)@right.T-weight))>1e-10:
        raise AssertionError("RHT roundtrip failed")
    states=np.arange(65536,dtype=np.int64)
    maps={"qtip_1mad":one_mad(states).astype(np.float64),
          "mul120":signed_byte(states*120).astype(np.float64)}
    maps={name:values/np.sqrt(np.mean(values**2)) for name,values in maps.items()}
    rows=[];begin=time.perf_counter()
    for name,mapping in maps.items():
        for rht in (False,True):
            W=left@weight@right if rht else weight
            Hr=right.T@H@right if rht else H
            scale=float(np.float32(np.sqrt(np.mean(W**2))))
            for ldlq in (False,True):
                reconstructed,tiles=quantize_blocks(W,Hr,mapping,scale,ldlq)
                sign_payload=(np.packbits(np.r_[left_signs>0,right_signs>0].astype(np.uint8),
                                          bitorder="big").tobytes() if rht else b"")
                payload=np.float32(scale).tobytes()+sign_payload+tiles
                (out/f"{name}_rht{int(rht)}_ldl{int(ldlq)}.bin").write_bytes(payload)
                stored_signs=np.unpackbits(np.frombuffer(sign_payload,dtype=np.uint8),bitorder="big") if rht else None
                if rht and not np.array_equal(stored_signs,np.r_[left_signs>0,right_signs>0]):
                    raise AssertionError("RHT signs did not roundtrip")
                decoded_scale=float(np.frombuffer(payload[:4],dtype=np.float32)[0])
                decoded=np.zeros_like(W)
                offset=4+len(sign_payload)
                for start in reversed(range(0,W.shape[1],16)):
                    path=unpack_path(payload[offset:offset+64],256)
                    decoded[:,start:start+16]=(mapping[path]*decoded_scale).reshape(16,16)
                    offset+=64
                decoder_error=float(np.max(np.abs(decoded-reconstructed)))
                if decoder_error>1e-6 or offset!=len(payload):
                    raise AssertionError((decoder_error,offset,len(payload)))
                restored=left.T@reconstructed@right.T if rht else reconstructed
                error=xval@(restored-weight).T
                nmse=float(np.sum(error**2)/np.sum(teacher**2))
                weight_nmse=float(np.sum((restored-weight)**2)/np.sum(weight**2))
                bits=len(payload)*8
                row=dict(generator=name,rht=rht,block_ldlq=ldlq,
                    output_nmse=nmse,weight_nmse=weight_nmse,
                    scale=scale,stored_bits=bits,stored_bpw=bits/4096,
                    decoder_max_abs=decoder_error,
                    state_bits=16,stride=2,tail_biting=True,
                    calibration_tokens=len(xcal),validation_tokens=len(xval),
                    elapsed_s=time.perf_counter()-begin)
                rows.append(row)
                with (out/"metrics.csv").open("w",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
                print(name,rht,ldlq,nmse,flush=True)
    (out/"config.json").write_text(json.dumps(dict(model="SmolLM2-135M",layer=15,
        projection="down_proj",output_rows=[0,15],input_columns=[0,255],
        calibration_token_offsets=[0,511],validation_token_offsets=[512,1023],
        transform_seed=20261127,block_size=16,trellis_length=256,
        rate=2,normalization="codebook RMS 1, matrix RMS frozen",
        note="Local submatrix functional ablation; no full-model perplexity or QTIP kernel export"),indent=2)+"\n")


if __name__=="__main__":run()
