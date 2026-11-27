"""R0 identity serialization/reload and evaluator check on fixed token IDs."""

import hashlib
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from experiments.evaluation.seedlm_model import load_model, perplexity, token_ids


def main():
    out = Path("experiments/results/reference/identity")
    out.mkdir(parents=True, exist_ok=True)
    assets = Path("data/smollm2")
    model = load_model(Path("data/smollm2-135m.safetensors"), assets, torch.bfloat16)
    tokenizer = Tokenizer.from_file(str(assets/"tokenizer.json"))
    tokens = token_ids(assets, "validation", tokenizer)[:128]
    state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/"identity.safetensors"
        save_file(state, str(path))
        reloaded = load_file(str(path))
        equal = all(torch.equal(tensor, reloaded[name]) for name, tensor in state.items())
        byte_count = path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with torch.no_grad():
        original_logits = model(tokens[None], use_cache=False).logits.float()
        original_nll, count = perplexity(model, tokens, 128, 1)
        model.load_state_dict(reloaded, strict=True)
        restored_logits = model(tokens[None], use_cache=False).logits.float()
        restored_nll, restored_count = perplexity(model, tokens, 128, 1)
    result = dict(tensor_equal=equal, state_tensors=len(state), serialized_bytes=byte_count,
       artifact_sha256=digest, max_logit_diff=float((original_logits-restored_logits).abs().max()),
       original_nll=original_nll, restored_nll=restored_nll, valid_targets=count,
       restored_targets=restored_count, context=128, dtype="bfloat16")
    (out/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    assert equal and result["max_logit_diff"] == 0 and original_nll == restored_nll


if __name__ == "__main__":
    main()
