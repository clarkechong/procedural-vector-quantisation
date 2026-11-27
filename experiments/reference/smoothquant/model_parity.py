"""SmolLM2 adaptation: full-network unquantized parity of upstream smoothing."""

import json
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

from experiments.evaluation.seedlm_model import load_model, token_ids


def main():
    source=Path("/tmp/smoothquant-official")
    if not source.exists():
        raise FileNotFoundError("clone pinned upstream SmoothQuant into /tmp/smoothquant-official")
    sys.path.insert(0,str(source))
    from smoothquant.smooth import smooth_lm
    out=Path("results/e04-07-reference-optimization/reference/smoothquant-model-parity")
    out.mkdir(parents=True,exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    assets=Path("data/smollm2")
    model=load_model(Path("data/smollm2-135m.safetensors"),assets,torch.float32).to(device)
    tokenizer=Tokenizer.from_file(str(assets/"tokenizer.json"))
    tokens=token_ids(assets,"validation",tokenizer)
    calibration=tokens[:128].to(device)
    evaluation=tokens[128:256].to(device)
    (out/"manifest.json").write_text(json.dumps(dict(calibration_offsets=[0,127],
        evaluation_offsets=[128,255], split="validation", alpha=.85,
        note="SmolLM2 adaptation, not Llama-2-7B reference"),indent=2)+"\n")
    scales={}
    hooks=[]
    for layer in range(model.config.num_hidden_layers):
        for projection in ("self_attn.q_proj","mlp.gate_proj"):
            module=model.model.layers[layer]
            for part in projection.split("."):
                module=getattr(module,part)
            name=f"model.layers.{layer}.{projection}"
            def hook(_,args,key=name):
                values=args[0].detach().abs().amax(dim=(0,1)).float().cpu()
                scales[key]=torch.maximum(scales[key],values) if key in scales else values
            hooks.append(module.register_forward_pre_hook(hook))
    with torch.no_grad():
        model(calibration[None],use_cache=False)
        original=model(evaluation[None],use_cache=False).logits.float()
    for hook in hooks:hook.remove()
    smooth_lm(model,scales,.85)
    with torch.no_grad():
        transformed=model(evaluation[None],use_cache=False).logits.float()
    difference=(original-transformed).abs()
    result=dict(model="SmolLM2-135M",dtype="float32",device=device,
       activation_scale_arrays=len(scales),calibration_tokens=128,evaluation_tokens=128,
       max_logit_diff=float(difference.max()),mean_logit_diff=float(difference.mean()),
       original_nll=float(torch.nn.functional.cross_entropy(original[:,:-1].reshape(-1,original.shape[-1]),evaluation[1:])),
       transformed_nll=float(torch.nn.functional.cross_entropy(transformed[:,:-1].reshape(-1,transformed.shape[-1]),evaluation[1:])))
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
