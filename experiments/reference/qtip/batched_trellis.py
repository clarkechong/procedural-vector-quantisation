"""Batched QTIP lookup-free two-pass trellis encoding.

This follows the pinned upstream V=1 transition and midpoint overlap rule.
The state history is stored as predecessor choice, not a dense transition matrix.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.reference.qtip.codebooks import one_mad, three_inst
from experiments.reference.qtip.tailbite import quantize as cpu_quantize


@torch.no_grad()
def viterbi(rows, mapping, overlap=None, state_bits=16, stride=2):
    if rows.ndim != 2:
        raise ValueError("rows must have shape [batch, length]")
    batch, length = rows.shape
    states = 1 << state_bits
    groups = states >> stride
    if mapping.numel() != states:
        raise ValueError("codebook size mismatch")
    target = rows.to(torch.float16).float()
    values = mapping.float()
    cost = (values[None] - target[:, 0, None]).square()
    device = rows.device
    if overlap is not None:
        allowed = (overlap.long() << stride)[:, None] + torch.arange(1 << stride, device=device)
        mask = torch.full_like(cost, torch.inf)
        mask.scatter_(1, allowed, 0)
        cost += mask
    # For each successor group, predecessors are group + candidate*groups.
    # int8 saves 8x history space versus upstream's int64 predecessor states.
    history = torch.empty((length - 1, batch, groups), dtype=torch.uint8, device=device)
    for position in range(1, length):
        predecessors = cost.reshape(batch, 1 << stride, groups)
        best_cost, choice = predecessors.min(dim=1)
        history[position - 1] = choice.to(torch.uint8)
        cost = best_cost.repeat_interleave(1 << stride, dim=1)
        cost += (values[None] - target[:, position, None]).square()
    if overlap is not None:
        allowed = overlap.long()[:, None] + (torch.arange(1 << stride, device=device) * groups)[None]
        mask = torch.full_like(cost, torch.inf)
        mask.scatter_(1, allowed, 0)
        cost += mask
    state = cost.argmin(dim=1)
    batch_index = torch.arange(batch, device=device)
    path = torch.empty((batch, length), dtype=torch.int32, device=device)
    path[:, -1] = state
    for position in range(length - 2, -1, -1):
        group = state >> stride
        state = group + history[position, batch_index, group].long() * groups
        path[:, position] = state
    return path


@torch.no_grad()
def quantize(rows, mapping, state_bits=16, stride=2):
    length = rows.shape[1]
    if length % 2:
        raise ValueError("QTIP two-pass midpoint requires even tile length")
    first = viterbi(rows.roll(length // 2, dims=1), mapping, state_bits=state_bits, stride=stride)
    overlap = first[:, length // 2].long() >> stride
    path = viterbi(rows, mapping, overlap, state_bits, stride)
    lowmask = (1 << (state_bits - stride)) - 1
    if not bool(torch.all((path[:, 1:] >> stride) == (path[:, :-1] & lowmask))):
        raise AssertionError("invalid path transition")
    if not bool(torch.all((path[:, 0] >> stride) == (path[:, -1] & lowmask))):
        raise AssertionError("path does not close")
    return mapping[path.long()], path


def run(args):
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    rows = rng.standard_normal((args.batch, args.length)).astype(np.float32)
    mapping_np = (one_mad if args.map == "1mad" else three_inst)(np.arange(65536)).astype(np.float32)
    if args.normalize:
        mapping_np /= np.sqrt(np.mean(mapping_np.astype(np.float64) ** 2))
    mapping = torch.from_numpy(mapping_np).to(device)
    data = torch.from_numpy(rows).to(device)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    start = time.perf_counter()
    decoded, path = quantize(data, mapping)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    path_np = path.cpu().numpy()
    decoded_np = decoded.cpu().numpy()
    mismatched = 0
    max_excess_sse = 0.0
    for index in range(args.check):
        cpu_decoded, cpu_path = cpu_quantize(rows[index], mapping_np)
        mismatched += int(not np.array_equal(path_np[index], cpu_path))
        cpu_sse = float(np.sum((cpu_decoded - rows[index].astype(np.float16).astype(np.float32)) ** 2))
        gpu_sse = float(np.sum((decoded_np[index] - rows[index].astype(np.float16).astype(np.float32)) ** 2))
        max_excess_sse = max(max_excess_sse, gpu_sse - cpu_sse)
    result = dict(device=str(device), gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  map=args.map, batch=args.batch, length=args.length, state_bits=16, stride=2,
                  normalized=args.normalize, seed=args.seed, elapsed_s=elapsed,
                  tiles_per_second=args.batch / elapsed, exact_path_mismatches=mismatched,
                  checked_tiles=args.check, max_excess_sse=max_excess_sse,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                  source_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",
                  note="Independent CPU parity; not direct upstream CUDA kernel parity or full projection")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if mismatched or max_excess_sse > 1e-4:
        raise AssertionError("batched encoder differs from CPU reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", choices=["1mad", "3inst"], default="1mad")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--check", type=int, default=8)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/e19-21-qtip-training/qtip_batched_1mad.json"))
    arguments = parser.parse_args()
    if arguments.check > arguments.batch:
        parser.error("--check cannot exceed --batch")
    run(arguments)
