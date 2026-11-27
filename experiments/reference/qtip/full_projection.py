"""Full SmolLM2 projection QTIP-map BlockLDLQ feasibility gate.

The initial run uses an unrotated q projection to isolate trellis/feedback/export.
It is an ablation, not the published QTIP RHT configuration.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from experiments.evaluation.seedlm_model import load_model, token_ids
from experiments.reference.qtip.batched_trellis import quantize
from experiments.reference.qtip.codebooks import one_mad, three_inst
from experiments.reference.qtip.pipeline_components import block_ldl
from experiments.reference.qtip.tailbite import pack_path, unpack_path
from experiments.synthesis.trellis import signed_byte


PERMUTE = torch.arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0, 4).flatten()
INV_PERMUTE = torch.argsort(PERMUTE)


def encode(weight, hessian, mapping, scale, device, compensate=True):
    m, n = weight.shape
    if m % 16 or n % 16:
        raise ValueError("full projection must be divisible into 16x16 tiles")
    regularized = hessian + torch.eye(n, dtype=hessian.dtype) * max(float(hessian.diag().mean()) * .01, 1e-12)
    L, _ = block_ldl(regularized, 16)
    L = L.to(device, dtype=torch.float32)
    W = weight.to(device, dtype=torch.float32)
    reconstructed = torch.zeros_like(W)
    map_device = (mapping * scale).to(device, dtype=torch.float32)
    payloads = []
    permute = PERMUTE.to(device)
    inverse = INV_PERMUTE.to(device)
    for start in reversed(range(0, n, 16)):
        target = W[:, start:start + 16].clone()
        if compensate and start + 16 < n:
            target += (W[:, start + 16:] - reconstructed[:, start + 16:]) @ L[start + 16:, start:start + 16]
        tiles = target.reshape(m // 16, 256)[:, permute]
        decoded, path = quantize(tiles, map_device)
        reconstructed[:, start:start + 16] = decoded[:, inverse].reshape(m, 16)
        payloads.extend((start, row, pack_path(p.cpu().numpy())) for row, p in enumerate(path))
    return reconstructed, payloads, regularized


def decode_payload(payload, shape, mapping):
    m, n = shape
    decoded = np.empty((m, n), dtype=np.float32)
    inverse = INV_PERMUTE.numpy()
    expected_size = 4 + (m * n // 256) * 64
    if len(payload) != expected_size:
        raise ValueError(f"payload length {len(payload)} differs from {expected_size}")
    scale = float(np.frombuffer(payload[:4], dtype=np.float32)[0])
    offset = 4
    for start in reversed(range(0, n, 16)):
        for row in range(m // 16):
            path = unpack_path(payload[offset:offset + 64], 256)
            offset += 64
            decoded[row * 16:(row + 1) * 16, start:start + 16] = (mapping[path] * scale)[inverse].reshape(16, 16)
    return decoded


def run(args):
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    data = Path("data/smollm2")
    tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
    tokens = token_ids(data, "validation", tokenizer)[:args.tokens]
    model = load_model(Path("data/smollm2-135m.safetensors"), data, torch.float32).to(device)
    module = model.model.layers[args.layer].self_attn.q_proj
    weight = module.weight.detach().double().cpu()
    captured = []
    def hook(_, inputs):
        captured.append(inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]).cpu())
    handle = module.register_forward_pre_hook(hook)
    with torch.no_grad():
        for block in tokens.split(128):
            model(block[None].to(device), use_cache=False)
    handle.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    X = torch.cat(captured).double()
    half = len(X) // 2
    calibration, validation = X[:half], X[half:]
    H = calibration.T @ calibration / len(calibration)
    states = np.arange(65536)
    map_np = (one_mad(states) if args.map == "1mad" else
              three_inst(states) if args.map == "3inst" else
              signed_byte(states * 120)).astype(np.float32)
    if args.normalize:
        map_np /= np.sqrt(np.mean(map_np.astype(np.float64) ** 2))
    mapping = torch.from_numpy(map_np)
    scale = float(np.float32(weight.square().mean().sqrt()))
    started = time.perf_counter()
    decoded, payloads, regularized = encode(weight, H, mapping, scale, device, compensate=args.ldlq)
    if device.type == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter() - started
    payload = np.float32(scale).tobytes() + b"".join(tile for _, _, tile in payloads)
    independent = decode_payload(payload, tuple(weight.shape), map_np)
    error = float(np.max(np.abs(independent - decoded.cpu().numpy())))
    if error > 1e-6:
        raise AssertionError(f"independent payload decoder mismatch: {error}")
    diff = decoded.double().cpu() - weight
    output_error = validation @ diff.T
    reference_output = validation @ weight.T
    result = dict(model="SmolLM2-135M", layer=args.layer, projection="q_proj", shape=list(weight.shape),
                  map=args.map, normalized_map=args.normalize, rht=False, block_ldlq=args.ldlq,
                  calibration_tokens=len(calibration), validation_tokens=len(validation),
                  input_token_offsets=[0, args.tokens - 1], scale=scale,
                  regularization=float((regularized - H).diag().mean()),
                  weight_nmse=float(diff.square().sum() / weight.square().sum()),
                  heldout_output_nmse=float(output_error.square().sum() / reference_output.square().sum()),
                  payload_bytes=len(payload) - 4, payload_bpw=8 * (len(payload) - 4) / weight.numel(),
                  scale_bytes=4, independent_decode_max_abs=error, encode_s=duration,
                  tiles=len(payloads), tiles_per_second=len(payloads) / duration,
                  note="Unrotated full projection BlockLDLQ ablation. Not native QTIP RHT or model PPL.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".bin").write_bytes(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", choices=["1mad", "3inst", "mul120"], default="1mad")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--ldlq", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/e19-21-qtip-training/qtip_full_qproj.json"))
    run(parser.parse_args())
