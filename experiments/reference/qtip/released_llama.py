"""Decode released QTIP Llama-2 packed weights for ROCm dense inference."""

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

from experiments.evaluation.seedlm_model import digest, load_model, perplexity, token_ids
from experiments.manifests.weights import PROJECTIONS


PINNED_SOURCE = Path("/tmp/qtip-official/lib/utils/matmul_had.py")
PINNED_DECOMPRESS = Path("/tmp/qtip-official/lib/utils/kernel_decompress.py")
PINNED_CODEBOOK = Path("/tmp/qtip-official/lib/codebook/bitshift.py")


def upstream_hadamard():
    tree = ast.parse(PINNED_SOURCE.read_text())
    names = {"get_hadK", "get_had172", "is_pow2", "matmul_hadU"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, "utils": SimpleNamespace(clean=lambda: None)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PINNED_SOURCE), "exec"), namespace)
    return namespace["matmul_hadU"]


def upstream_codec():
    tree = ast.parse(PINNED_DECOMPRESS.read_text())
    decode = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "decode_compressed")
    decode.decorator_list = []  # Eager execution; the body is unchanged.
    book = ast.parse(PINNED_CODEBOOK.read_text())
    quantlut = next(node for node in book.body
                    if isinstance(node, ast.FunctionDef) and node.name == "quantlut_sym")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[decode, quantlut], type_ignores=[]),
                 str(PINNED_DECOMPRESS), "exec"), namespace)
    return namespace["decode_compressed"], namespace["quantlut_sym"]


def unpack_states(packed):
    """Invert pinned bitshift_codebook.pack_trellis for L16,K2,V2 tiles."""
    words = np.asarray(packed, dtype=np.int16).view(np.uint16)
    assert words.ndim == 2 and words.shape[1] == 32
    bits = ((words[:, :, None] >> np.arange(15, -1, -1, dtype=np.uint16)) & 1)
    bits = bits.reshape(len(words), 512)
    bits = np.concatenate((bits, bits[:, :12]), axis=1)
    states = np.empty((len(words), 128), dtype=np.uint16)
    states[:, 0] = bits[:, :16] @ (1 << np.arange(15, -1, -1, dtype=np.uint32))
    weights = np.array([8, 4, 2, 1], dtype=np.uint16)
    for i in range(1, 128):
        nibble = bits[:, 16 + 4 * (i - 1):20 + 4 * (i - 1)] @ weights
        states[:, i] = ((states[:, i - 1].astype(np.uint32) << 4) & 65535) + nibble
    return states


def decode_trellis(packed, tlut, m, n, chunk_tiles=2048):
    assert m % 16 == n % 16 == 0
    tiles = (m // 16) * (n // 16)
    assert packed.shape == (tiles, 32) and tlut.shape == (512, 2)
    flat = np.empty((tiles, 256), dtype=np.float16)
    for start in range(0, tiles, chunk_tiles):
        end = min(start + chunk_tiles, tiles)
        states = unpack_states(packed[start:end]).astype(np.uint64)
        mixed = (states + 1) * states
        indices = ((mixed >> 6) & 511).astype(np.intp)
        sign = (1 - 2 * ((mixed >> 15) & 1).astype(np.int8))
        values = tlut[indices].copy()
        values[:, :, 0] *= sign
        flat[start:end] = values.reshape(end - start, 256)
    return flat.reshape(m // 16, n // 16, 16, 16).transpose(0, 2, 1, 3).reshape(m, n)


class DecodedQTIPLinear(torch.nn.Module):
    def __init__(self, hat_weight, su, sv):
        super().__init__()
        self.register_buffer("hat_weight", hat_weight)
        self.register_buffer("su", su)
        self.register_buffer("sv", sv)

    def forward(self, value):
        # Mirrors upstream BitshiftLinear.forward(mode='train-fixW').
        shape = value.shape
        x = value.reshape(-1, shape[-1]).float() * self.su
        x = (x.to(self.hat_weight.dtype) @ self.hat_weight.T).float()
        x = x * (self.sv * 32)
        return x.reshape(*shape[:-1], -1).to(value.dtype)


def decode_projection(tensors, key, m, n, transform, codec, quantlut, device):
    packed = tensors.get_tensor(key + ".trellis").to(device)
    tlut = tensors.get_tensor(key + ".tlut").to(device)
    lut = quantlut(tlut, 16, 9)
    raw = codec(16, 9, 2, 1, m, n, packed.reshape(-1), lut)
    weight = raw.float() / 32
    weight = transform(transform(weight).T).T.contiguous().to(torch.float16)
    su = tensors.get_tensor(key + ".SU").to(device)
    sv = tensors.get_tensor(key + ".SV").to(device)
    return DecodedQTIPLinear(weight, su, sv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/llama-2-7b-hf"))
    parser.add_argument("--quantized", type=Path, default=Path("data/qtip-llama-2-7b/model.safetensors"))
    parser.add_argument("--data", type=Path, default=Path("data/wikitext-2-raw-v1"))
    parser.add_argument("--out", type=Path, default=Path("runs/llama2-7b/qtip-released"))
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--windows", type=int, default=83)
    parser.add_argument("--layers", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model = load_model(args.base, args.base, torch.float16)
    tokenizer = Tokenizer.from_file(str(args.base / "tokenizer.json"))
    tokens = token_ids(args.data, "test", tokenizer, model.config.bos_token_id)
    tokens = tokens[:args.context * args.windows]
    assert len(tokens) == args.context * args.windows
    transform = upstream_hadamard()
    codec, quantlut = upstream_codec()
    with safe_open(str(args.quantized), framework="pt", device="cpu") as tensors:
        projection_prefixes = tuple(
            f"model.layers.{layer}.{group}.{name}."
            for layer in range(32) for name, group in PROJECTIONS.items()
        )
        with torch.no_grad():
            state = model.state_dict()
            for key, target in state.items():
                if key.startswith(projection_prefixes):
                    continue
                if key not in tensors.keys():
                    raise KeyError(f"Released checkpoint is missing {key}")
                target.copy_(tensors.get_tensor(key))
        for layer in range(args.layers):
            for name, group in PROJECTIONS.items():
                parent = getattr(model.model.layers[layer], group)
                original = getattr(parent, name)
                key = f"model.layers.{layer}.{group}.{name}"
                decoded = decode_projection(tensors, key, *original.weight.shape,
                                            transform, codec, quantlut, device)
                setattr(parent, name, decoded)
            print(f"decoded layer {layer + 1}/{args.layers}", flush=True)
    model.to(device).eval()
    with torch.inference_mode():
        nll, scored = perplexity(model, tokens, args.context, 1, device)
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(nll=nll, perplexity=float(np.exp(nll)), scored_tokens=scored,
                  context=args.context, windows=args.windows, decoded_layers=args.layers,
                  checkpoint_sha256=digest(args.quantized),
                  base_index_sha256=digest(args.base / "model.safetensors.index.json"),
                  test_sha256=digest(args.data / "test-00000-of-00001.parquet"),
                  upstream_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",
                  note="Released HYB checkpoint, pinned PyTorch decompressor and Hadamard with local dense forward adapter; no native QTIP kernel")
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
