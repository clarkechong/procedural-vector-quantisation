"""Pack TinyStories QAT weights and independently reconstruct their logits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.training.tinystories_transformer import Transformer,QLinear


def export_one(folder,method,seed=11):
    if method.startswith(("ptq","finetune")):
        quant_method="procedural2" if method.endswith("2") else "procedural1"
    else:
        quant_method=method
    if quant_method=="dense": return None
    torch.manual_seed(seed)
    source=Transformer(quant_method)
    source.load_state_dict(torch.load(folder/f"{method}.pt",map_location="cpu",weights_only=True))
    qlayers={name:module for name,module in source.named_modules() if isinstance(module,QLinear)}
    payload=bytearray(); manifest=[]
    for name,tensor in source.named_parameters():
        if name.endswith(".latent_logits"):
            continue
        if name.endswith(".weight") and name.removesuffix(".weight") in qlayers:
            layer=qlayers[name.removesuffix(".weight")]
            values=tensor.detach().numpy()
            scale_offset=len(payload)
            payload.extend(np.float32(layer.scale).tobytes())
            if quant_method=="scalar2":
                levels=np.array([-3.,-1.,1.,3.],dtype=np.float32)/np.sqrt(5)
                indices=np.abs(values[...,None]/layer.scale-levels).argmin(-1).astype(np.uint8)
                bits=np.stack((indices&1,indices>>1),axis=-1).reshape(-1)
                data=np.packbits(bits,bitorder="little").tobytes()
                kind="scalar2"
            elif quant_method=="latent2":
                logits=layer.latent_logits.detach().numpy()
                indices=((logits>0).astype(np.uint16)*(1<<np.arange(8))).sum(-1).astype(np.uint8)
                data=indices.tobytes()
                kind="procedural"
            else:
                book=layer.book.numpy()
                width=layer.width
                tiles=values.reshape(-1,width)/layer.scale
                scores=(book**2).sum(-1)[None]-2*tiles@book.T
                data=scores.argmin(-1).astype(np.uint8).tobytes()
                kind="procedural"
            offset=len(payload); payload.extend(data)
            manifest.append(dict(name=name,kind=kind,shape=list(values.shape),offset=offset,
                                 bytes=len(data),scale_offset=scale_offset,width=layer.width))
        else:
            data=tensor.detach().numpy().astype("<f4").tobytes()
            offset=len(payload);payload.extend(data)
            manifest.append(dict(name=name,kind="float32",shape=list(tensor.shape),
                                 offset=offset,bytes=len(data)))
    data=bytes(payload)
    (folder/f"{method}.bin").write_bytes(data)
    (folder/f"{method}_payload.json").write_text(json.dumps(manifest,indent=2)+"\n")
    # Independent parse of the payload into a dense model.
    torch.manual_seed(seed)
    restored=Transformer("dense")
    state=restored.state_dict()
    for entry in manifest:
        raw=data[entry["offset"]:entry["offset"]+entry["bytes"]]
        shape=entry["shape"]
        scale=(np.frombuffer(data[entry["scale_offset"]:entry["scale_offset"]+4],dtype="<f4")[0]
               if "scale_offset" in entry else None)
        if entry["kind"]=="float32":
            array=np.frombuffer(raw,dtype="<f4").copy().reshape(shape)
        elif entry["kind"]=="scalar2":
            count=int(np.prod(shape))
            bits=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder="little")[:2*count].reshape(-1,2)
            indices=bits[:,0]+2*bits[:,1]
            levels=np.array([-3.,-1.,1.,3.],dtype=np.float32)/np.sqrt(5)
            array=(levels[indices].reshape(shape)*scale).astype(np.float32)
        else:
            layer=qlayers[entry["name"].removesuffix(".weight")]
            indices=np.frombuffer(raw,dtype=np.uint8)
            array=(layer.book.numpy()[indices].reshape(shape)*scale).astype(np.float32)
        state[entry["name"]]=torch.from_numpy(array.copy())
    restored.load_state_dict(state)
    sample=torch.randint(0,256,(2,32),generator=torch.Generator().manual_seed(123))
    with torch.no_grad():
        difference=(source(sample)-restored(sample)).abs().max().item()
    expected=sum(entry["bytes"]+int("scale_offset" in entry)*4 for entry in manifest)
    if len(data)!=expected or difference>1e-5:
        raise AssertionError((len(data),expected,difference))
    return dict(method=method,bytes=len(data),bits=len(data)*8,
                max_logit_difference=difference,parameters=sum(p.numel() for p in restored.parameters()))


def run(folder):
    config=json.loads((folder/"config.json").read_text())
    results=[]
    for method in config["methods"]:
        if method!="dense":
            row=export_one(folder,method,config["seed"])
            results.append(row)
            print(row,flush=True)
    (folder/"export_validation.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    run(parser.parse_args().folder)
