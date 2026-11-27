"""QTIP Hessian, RHT, and block LDL numerical component checks."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import hadamard


PINNED_MATH = Path("/tmp/qtip-official/lib/utils/math_utils.py")


def hessian(activations):
    x = torch.as_tensor(activations, dtype=torch.float64)
    return x.T @ x / len(x)


def randomized_hadamard(matrix, signs):
    n = matrix.shape[-1]
    if n & (n-1):
        raise ValueError("This component fixture uses power-of-two dimensions")
    transform = torch.as_tensor(hadamard(n).copy(), dtype=matrix.dtype,
                                device=matrix.device) / np.sqrt(n)
    return (matrix * signs) @ transform


def block_ldl(H, block):
    n = len(H)
    if n % block:
        raise ValueError("block must divide dimension")
    Lchol = torch.linalg.cholesky(H)
    result = Lchol.clone()
    blocks = []
    for start in range(0, n, block):
        diagonal = Lchol[start:start+block, start:start+block]
        blocks.append(diagonal @ diagonal.T)
        result[:, start:start+block] = Lchol[:, start:start+block] @ torch.linalg.inv(diagonal)
        result[start:start+block,start:start+block] = torch.eye(block,dtype=H.dtype,device=H.device)
    return result, torch.block_diag(*blocks)


def run(out=Path("results/e04-07-reference-optimization/reference/qtip-pipeline-components")):
    out.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator().manual_seed(20261127)
    x=torch.randn((512,64),generator=generator,dtype=torch.float64)
    H=hessian(x)
    direct=sum(torch.outer(row,row) for row in x)/len(x)
    signs=torch.where(torch.rand(64,generator=generator)>0.5,1.,-1.).double()
    transformed=randomized_hadamard(x,signs)
    transformed_H=hessian(transformed)
    transform=torch.diag(signs) @ (torch.as_tensor(hadamard(64).copy()).double()/8)
    expected_H=transform.T @ H @ transform
    regularized=H+torch.eye(64,dtype=torch.float64)*0.1
    L,D=block_ldl(regularized,16)
    spec=importlib.util.spec_from_file_location("pinned_qtip_math",PINNED_MATH)
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    upstream_L,upstream_blocks=upstream.block_LDL(regularized,16)
    upstream_D=torch.block_diag(*upstream_blocks)
    metrics=dict(hessian_max_abs=float((H-direct).abs().max()),
        rht_inverse_max_abs=float((transformed @ transform.T-x).abs().max()),
        transformed_hessian_max_abs=float((transformed_H-expected_H).abs().max()),
        block_ldl_reconstruction_max_abs=float((L@D@L.T-regularized).abs().max()),
        upstream_L_max_abs=float((L-upstream_L).abs().max()),
        upstream_D_max_abs=float((D-upstream_D).abs().max()),
        dimension=64,block=16,calibration_vectors=512,
        upstream_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    run()
