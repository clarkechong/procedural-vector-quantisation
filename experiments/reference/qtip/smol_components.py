"""SmolLM2 calibration Hessian and pinned QTIP RHT/block-LDL component parity."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

from experiments.evaluation.seedlm_model import load_model,token_ids
from experiments.reference.qtip.pipeline_components import block_ldl,hessian


SOURCE=Path("/tmp/qtip-official/lib/utils/matmul_had.py")


def upstream_hadamard():
    tree=ast.parse(SOURCE.read_text())
    selected={"get_hadK","get_had12","get_had36","is_pow2","matmul_hadU"}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in selected]
    namespace={"torch":torch,"utils":SimpleNamespace(clean=lambda:None)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),"exec"),namespace)
    return namespace["matmul_hadU"]


def run(out=Path("results/e04-07-reference-optimization/reference/qtip-smollm-components")):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=Tokenizer.from_file("data/smollm2/tokenizer.json")
    tokens=token_ids(Path("data/smollm2"),"validation",tokenizer)[:512]
    model=load_model(Path("data/smollm2-135m.safetensors"),Path("data/smollm2"),torch.float32).to(device)
    module=model.model.layers[15].mlp.down_proj
    captured=[]
    def hook(_,args):captured.append(args[0].detach().float().reshape(-1,args[0].shape[-1]))
    handle=module.register_forward_pre_hook(hook)
    with torch.no_grad():
        for block in tokens.split(128):model(block[None].to(device),use_cache=False)
    handle.remove();del model
    x=torch.cat(captured).to(torch.float64)
    H=hessian(x)
    n=H.shape[0]
    transform=upstream_hadamard()
    torch.manual_seed(20261127)
    signs=torch.where(torch.rand(n,device=device)>0.5,1.,-1.).to(torch.float64)
    # Upstream function is evaluated in FP64 for this reference component check.
    rht=transform(x*signs)
    U=transform(torch.eye(n,device=device,dtype=torch.float64))
    explicit=(x*signs)@U
    H_from_activations=hessian(rht)
    H_from_matrix=U.T@(signs[:,None]*H*signs[None,:])@U
    diagmean=torch.diag(H_from_activations).mean()
    regularized=H_from_activations+torch.eye(n,device=device,dtype=torch.float64)*(0.01*diagmean)
    L,D=block_ldl(regularized,16)
    metrics=dict(dimension=n,calibration_tokens=len(tokens),layer=15,projection="down_proj",
        hessian_diag_mean=float(H.diag().mean()),rht_max_abs=float((rht-explicit).abs().max()),
        rht_orthogonality_max_abs=float((U.T@U-torch.eye(n,device=device,dtype=torch.float64)).abs().max()),
        transformed_hessian_max_abs=float((H_from_activations-H_from_matrix).abs().max()),
        block_ldl_reconstruction_max_abs=float((L@D@L.T-regularized).abs().max()),
        activation_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
        upstream_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",
        note="Actual SmolLM2 activation Hessian, pinned upstream RHT source extracted without CUDA imports; no BlockLDLQ or full-model QTIP quantization")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":run()
