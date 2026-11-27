"""Independently pack and reload 2-bpw MLP endpoints from the landscape run."""

import argparse
import json
from pathlib import Path
import struct

import numpy as np
import torch

from experiments.training.landscape_mlp import load_layers
from experiments.training.low_rate_qat import make_book
from experiments.training.mnist_qat import MLP, data_split


SHAPES = ((256,784), (128,256), (10,128))
MAGIC = b"MLPQ"


def code_indices(weight, scale, method):
    weight = np.asarray(weight, dtype=np.float32)
    if method == "scalar2":
        levels = np.asarray([-3,-1,1,3], dtype=np.float32)/np.sqrt(np.float32(5))
        return np.abs(weight[..., None]/scale-levels).argmin(-1).astype(np.uint8).ravel()
    if method == "mul120_2bpw":
        book = make_book("mul120",4)
        tiles = weight.reshape(-1,4)/scale
        norms = np.sum(book*book,axis=1)
        result = np.empty(len(tiles),dtype=np.uint8)
        for first in range(0,len(tiles),4096):
            batch=tiles[first:first+4096]
            score=norms[None]-2*batch@book.T
            result[first:first+len(batch)]=score.argmin(-1)
        return result
    raise ValueError(method)


def pack(state, scales, method):
    codec = 1 if method == "scalar2" else 2
    payload = bytearray(MAGIC + struct.pack("<BB",1,codec))
    for i,shape in enumerate(SHAPES):
        weight=state[f"layers.{i}.weight"].cpu().numpy()
        bias=state[f"layers.{i}.bias"].cpu().numpy()
        assert weight.shape==shape and bias.shape==(shape[0],)
        codes=code_indices(weight,scales[i],method)
        if method=="scalar2":
            assert len(codes)%4==0
            packed=(codes[0::4] | (codes[1::4]<<2) |
                    (codes[2::4]<<4) | (codes[3::4]<<6)).tobytes()
        else:
            packed=codes.tobytes()
        payload.extend(struct.pack("<f",scales[i]))
        payload.extend(np.asarray(bias,dtype="<f4").tobytes())
        payload.extend(packed)
    return bytes(payload)


def unpack(payload):
    if payload[:4]!=MAGIC or payload[4]!=1 or payload[5] not in (1,2):
        raise ValueError("unknown packed MLP format")
    method="scalar2" if payload[5]==1 else "mul120_2bpw"
    cursor=6;state={}
    for i,(rows,cols) in enumerate(SHAPES):
        scale=struct.unpack_from("<f",payload,cursor)[0];cursor+=4
        bias=np.frombuffer(payload,dtype="<f4",count=rows,offset=cursor).copy();cursor+=rows*4
        count=rows*cols
        if method=="scalar2":
            packed=np.frombuffer(payload,dtype=np.uint8,count=count//4,offset=cursor)
            codes=np.stack((packed&3,(packed>>2)&3,(packed>>4)&3,packed>>6),axis=-1).ravel()
            levels=np.asarray([-3,-1,1,3],dtype=np.float32)/np.sqrt(np.float32(5))
            weight=levels[codes].reshape(rows,cols)*np.float32(scale)
            cursor+=count//4
        else:
            codes=np.frombuffer(payload,dtype=np.uint8,count=count//4,offset=cursor)
            book=make_book("mul120",4)
            weight=book[codes].reshape(rows,cols)*np.float32(scale)
            cursor+=count//4
        state[f"layers.{i}.weight"]=torch.from_numpy(weight.copy())
        state[f"layers.{i}.bias"]=torch.from_numpy(bias)
    if cursor!=len(payload):
        raise ValueError("trailing or missing payload bytes")
    return state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--result",type=Path,default=Path("experiments/results/experiments-13-18/landscape_mlp"))
    args=parser.parse_args()
    result=args.result
    config=json.loads((result/"config.json").read_text())
    outputs=[]
    image,_,_,_,_,_,_=data_split()
    x=torch.as_tensor(image[:100].reshape(100,-1),dtype=torch.float32)/255
    for seed in config["seeds"]:
        run_dir=Path(config["checkpoint_dir"])/f"seed{seed}"
        initial=torch.load(run_dir/"initial.pt",weights_only=False,map_location="cpu")
        scales=[float(initial[f"layers.{i}.weight"].square().mean().sqrt()) for i in range(3)]
        for method in ("scalar2","mul120_2bpw"):
            saved=torch.load(run_dir/f"{method}_epoch{config['epochs']}.pt",weights_only=False,map_location="cpu")
            state=saved["decoded"]
            payload=pack(state,scales,method)
            path=run_dir/f"{method}.bin";path.write_bytes(payload)
            decoded=unpack(path.read_bytes())
            source=MLP();reload=MLP()
            load_layers(source,state);load_layers(reload,decoded)
            with torch.inference_mode():
                max_logit_error=float((source(x)-reload(x)).abs().max())
            max_weight_error=max(float((state[k].cpu()-decoded[k]).abs().max()) for k in decoded)
            outputs.append(dict(seed=seed,method=method,packed_bytes=len(payload),
                                projection_bpw=(len(payload)-6-3*4-394*4)*8/234752,
                                total_bits_per_original_weight=len(payload)*8/234752,
                                max_logit_error=max_logit_error,
                                max_weight_error=max_weight_error,
                                format="MLPQ v1, 6-byte header, FP32 scale per layer, FP32 biases, 2-bit scalar or 8-bit seed per four weights"))
    (result/"packed_endpoints.json").write_text(json.dumps(outputs,indent=2)+"\n")
    print(json.dumps(outputs,indent=2))


if __name__=="__main__":main()
