"""Small byte-level TinyStories transformer from-scratch QAT pilot."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from experiments.training.low_rate_qat import make_book
from experiments.training.latent_qat import HardSeed


TRAIN = Path("data/tinystories/data/train-00000-of-00004-2d5a1467fff1081b.parquet")
VALID = Path("data/tinystories/data/validation-00000-of-00001-869c898b519ad725.parquet")


class QLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, method):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.empty(output_size,input_size))
        self.bias=torch.nn.Parameter(torch.zeros(output_size))
        torch.nn.init.kaiming_uniform_(self.weight,a=5**0.5)
        self.scale=float(self.weight.detach().square().mean().sqrt())
        self.method=method
        if method.startswith("procedural") or method=="latent2":
            width={"procedural2":4,"procedural1":8,"procedural0.5":16}.get(method,4)
            self.width=width
            self.register_buffer("book",torch.tensor(make_book("mul120",width)))
            if method=="latent2":
                tiles=self.weight.detach().reshape(-1,width)/self.scale
                scores=self.book.square().sum(-1)[None]-2*tiles@self.book.T
                seed=scores.argmin(-1)
                bits=((seed[:,None]>>torch.arange(8))&1).float()
                self.latent_logits=torch.nn.Parameter((bits*2-1)*0.25)
                self.weight.requires_grad_(False)
        else:
            self.width=0
            self.register_buffer("book",None)

    def forward(self,x):
        w=self.weight
        if self.method=="latent2":
            w=HardSeed.apply(self.latent_logits,self.book,"discrete_jacobian").reshape_as(w)*self.scale
            return F.linear(x,w,self.bias)
        if self.method!="dense":
            with torch.no_grad():
                if self.method in ("scalar2","scalar4","binary"):
                    if self.method=="scalar2":
                        levels=torch.tensor([-3.,-1.,1.,3.],device=w.device)/np.sqrt(5)
                    elif self.method=="scalar4":
                        levels=torch.tensor(list(range(-15,16,2)),device=w.device,dtype=w.dtype)
                        levels=levels/levels.square().mean().sqrt()
                    else:
                        levels=torch.tensor([-1.,1.],device=w.device)
                    q=levels[(w[...,None]/self.scale-levels).abs().argmin(-1)]*self.scale
                else:
                    tiles=w.reshape(-1,self.width)/self.scale
                    scores=self.book.square().sum(-1)[None]-2*tiles@self.book.T
                    q=self.book[scores.argmin(-1)].reshape_as(w)*self.scale
            w=w+(q-w).detach()
        return F.linear(x,w,self.bias)


class Block(torch.nn.Module):
    def __init__(self,method):
        super().__init__()
        self.ln1=torch.nn.LayerNorm(128)
        self.qkv=QLinear(128,384,method)
        self.proj=QLinear(128,128,method)
        self.ln2=torch.nn.LayerNorm(128)
        self.up=QLinear(128,512,method)
        self.down=QLinear(512,128,method)

    def forward(self,x):
        batch,length,_=x.shape
        q,k,v=self.qkv(self.ln1(x)).reshape(batch,length,3,4,32).permute(2,0,3,1,4).unbind(0)
        x=x+self.proj(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(batch,length,128))
        return x+self.down(F.gelu(self.up(self.ln2(x))))


class Transformer(torch.nn.Module):
    def __init__(self,method):
        super().__init__()
        self.embedding=torch.nn.Embedding(256,128)
        self.position=torch.nn.Embedding(256,128)
        self.blocks=torch.nn.ModuleList([Block(method) for _ in range(4)])
        self.final_norm=torch.nn.LayerNorm(128)
        self.head=torch.nn.Linear(128,256,bias=False)

    def forward(self,tokens):
        length=tokens.shape[1]
        positions=torch.arange(length,device=tokens.device)
        x=self.embedding(tokens)+self.position(positions)
        for block in self.blocks:
            x=block(x)
        return self.head(self.final_norm(x))


def sequences(path,count,seed,context):
    column=pq.read_table(path,columns=["text"]).column(0)
    rng=np.random.default_rng(seed)
    chosen=rng.choice(len(column),count,replace=False)
    text=b"\0".join(column[int(i)].as_py().encode("utf-8",errors="replace") for i in chosen)
    n=(len(text)-1)//(context+1)
    data=np.frombuffer(text[:n*(context+1)],dtype=np.uint8).copy().reshape(n,context+1)
    return torch.tensor(data,dtype=torch.long),hashlib.sha256(chosen.tobytes()).hexdigest()


@torch.no_grad()
def evaluate(model,data,batches=64,batch_size=16):
    model.eval()
    losses=[]
    for start in range(min(batches*batch_size,len(data))//batch_size):
        batch=data[start*batch_size:(start+1)*batch_size]
        logits=model(batch[:,:-1])
        losses.append(float(F.cross_entropy(logits.reshape(-1,256),batch[:,1:].reshape(-1))))
    return float(np.mean(losses))


@torch.no_grad()
def generate(model,prompt="Once upon a time",tokens=160):
    device=next(model.parameters()).device
    stream=list(prompt.encode())
    model.eval()
    for _ in range(tokens):
        logits=model(torch.tensor(stream[-256:],device=device)[None])[:,-1]
        next_byte=int(torch.multinomial(F.softmax(logits/0.8,dim=-1),1))
        stream.append(next_byte)
    return bytes(stream).decode("utf-8",errors="replace")


def storage(model,method):
    procedural=sum(layer.weight.numel() for layer in model.modules() if isinstance(layer,QLinear))
    # Count all non-QLinear parameters by identity, avoiding name heuristics.
    qweights={id(layer.weight) for layer in model.modules() if isinstance(layer,QLinear)}
    latents={id(layer.latent_logits) for layer in model.modules()
             if isinstance(layer,QLinear) and hasattr(layer,"latent_logits")}
    other=sum(p.numel() for p in model.parameters() if id(p) not in qweights|latents)
    rate={"dense":32,"scalar4":4,"scalar2":2,"binary":1,
          "procedural2":2,"procedural1":1,"procedural0.5":0.5,"latent2":2}[method]
    scale_bits=0 if method=="dense" else 32*16
    return dict(procedural_weights=procedural,other_dense_parameters=other,
                procedural_payload_bits=procedural*rate+scale_bits,
                other_dense_bits=other*32,
                total_bits=procedural*rate+scale_bits+other*32,
                projection_bpw=rate+scale_bits/procedural)


def run(args):
    torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type!="cuda": raise RuntimeError("GPU required")
    train,train_hash=sequences(TRAIN,args.train_stories,20261130,256)
    valid,valid_hash=sequences(VALID,args.valid_stories,20261201,256)
    train=train.to(device);valid=valid.to(device)
    rows=[]; samples={}
    torch.manual_seed(args.seed)
    base=Transformer("dense")
    initial={k:v.detach().clone() for k,v in base.state_dict().items()}
    dense_state=None
    for method in args.methods:
        torch.manual_seed(args.seed)
        if method.startswith(("ptq", "finetune")):
            quant_method = "procedural2" if method.endswith("2") else "procedural1"
        else:
            quant_method = method
        model=Transformer(quant_method).to(device)
        source = dense_state if method.startswith(("ptq", "finetune")) else initial
        if source is None:
            raise ValueError("Dense training must precede compression controls")
        model.load_state_dict(source,strict=False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
        begin=time.perf_counter()
        steps = 0 if method.startswith("ptq") else args.finetune_steps if method.startswith("finetune") else args.steps
        for step in range(steps+1):
            if step%args.eval_every==0 or step==steps:
                ce=evaluate(model,valid)
                rows.append(dict(method=method,step=step,validation_ce=ce,
                                 validation_ppl=float(np.exp(ce)),elapsed_s=time.perf_counter()-begin))
                with (out/"metrics.csv").open("w",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
                print(method,step,ce,flush=True)
            if step==steps: break
            model.train()
            indices=torch.randint(0,len(train),(args.batch_size,),device=device)
            batch=train[indices]
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch[:,:-1])
            loss=F.cross_entropy(logits.reshape(-1,256),batch[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
        if method=="dense":
            dense_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        torch.save(model.state_dict(),out/f"{method}.pt")
        samples[method]=generate(model)
        (out/f"{method}_storage.json").write_text(json.dumps(storage(model,quant_method),indent=2)+"\n")
    (out/"samples.json").write_text(json.dumps(samples,indent=2)+"\n")
    (out/"config.json").write_text(json.dumps(dict(seed=args.seed,architecture=dict(layers=4,
         hidden=128,heads=4,ffn=512,context=256,tokenizer="UTF-8 byte",vocabulary=256),
         train_shard=str(TRAIN),validation_shard=str(VALID),train_stories=args.train_stories,
         valid_stories=args.valid_stories,train_indices_sha256=train_hash,
         valid_indices_sha256=valid_hash,train_sequences=len(train),validation_sequences=len(valid),
         methods=args.methods,steps=args.steps,finetune_steps=args.finetune_steps,
         batch_size=args.batch_size,lr=args.lr,
         optimizer="AdamW",device=str(device),
         note="One-seed bounded from-scratch pilot on one of four public training shards; no blind test"),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--methods",nargs="+",default=["dense","scalar2","procedural2","procedural1",
                                                   "ptq2","finetune2","ptq1","finetune1"])
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--steps",type=int,default=400)
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--finetune-steps",type=int,default=400)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--train-stories",type=int,default=20000)
    p.add_argument("--valid-stories",type=int,default=1000)
    p.add_argument("--lr",type=float,default=0.0003)
    p.add_argument("--out",default="experiments/results/procedural_training/tinystories_transformer")
    p.add_argument("--require-gpu",action="store_true")
    run(p.parse_args())
