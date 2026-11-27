"""Memory-bounded all-seed ROCm pilot for the frozen SeedLM coefficient rule.

The GPU scores every legal seed.  This is an encoder prototype, not a claim of
published model-quality reproduction; near ties require CPU reference reranking.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.encoding.paper_seedlm import PaperEncoder
from experiments.encoding.seedlm import PUBLISHED, decode, pack
from experiments.manifests.weights import Weights


class GPUPaperEncoder:
    def __init__(self, config, library, seed_tile=1024, block_tile=32):
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("requires ROCm GPU")
        self.cpu = PaperEncoder(config, 65535, library, threads=8)
        self.config = config
        self.seed_tile = seed_tile
        self.block_tile = block_tile
        self.inverse = torch.from_numpy(self.cpu.inverse).to("cuda")
        self.bases = torch.from_numpy(self.cpu.bases).to("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False

    @torch.inference_mode()
    def fields(self, source):
        source = np.asarray(source, dtype=np.float32)
        if source.ndim != 2 or source.shape[1] != self.config.block:
            raise ValueError("incorrect SeedLM block shape")
        all_seeds = np.empty(len(source), dtype=np.uint32)
        all_codes = np.empty((len(source), self.config.latent), dtype=np.int8)
        all_exponents = np.empty(len(source), dtype=np.int8)
        for first in range(0, len(source), self.block_tile):
            last = min(len(source), first+self.block_tile)
            x = torch.from_numpy(source[first:last]).to("cuda")
            best = torch.full((len(x),), float("inf"), device="cuda")
            chosen_seed = torch.zeros(len(x), device="cuda", dtype=torch.int32)
            chosen_code = torch.zeros((len(x), self.config.latent), device="cuda", dtype=torch.int8)
            chosen_exponent = torch.zeros(len(x), device="cuda", dtype=torch.int8)
            for start in range(0, 65535, self.seed_tile):
                stop = min(start+self.seed_tile, 65535)
                inverse = self.inverse[start:stop]
                bases = self.bases[start:stop]
                # B x S x P.  Transposed CPU arrays match the pinned C++ scorer.
                coefficients = torch.einsum("bc,spc->bsp", x, inverse)
                peak = coefficients.abs().amax(-1)
                _, powers = torch.frexp(peak)
                exponent = (powers-1).clamp(-8, 7)
                exponent = torch.where(peak > 0, exponent, -8)
                scale = torch.ldexp(torch.ones_like(peak), exponent)
                codes = torch.round(coefficients/scale[..., None]).clamp(-8, 7)
                reconstructed = torch.einsum("bsp,spc->bsc", codes*scale[..., None], bases)
                error = (x[:, None, :] - reconstructed).square().sum(-1)
                local_error, local_index = error.min(-1)
                improve = local_error < best
                row = torch.arange(len(x), device="cuda")
                best = torch.where(improve, local_error, best)
                chosen_seed = torch.where(improve, local_index.to(torch.int32)+start+1, chosen_seed)
                chosen_code = torch.where(improve[:, None],
                                          codes[row, local_index].to(torch.int8), chosen_code)
                chosen_exponent = torch.where(improve, exponent[row, local_index].to(torch.int8),
                                              chosen_exponent)
            all_seeds[first:last] = chosen_seed.cpu().numpy()
            all_codes[first:last] = chosen_code.cpu().numpy()
            all_exponents[first:last] = chosen_exponent.cpu().numpy()
        return all_seeds, all_codes, all_exponents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=[3, 4], default=4)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--seed-tile", type=int, default=1024)
    parser.add_argument("--block-tile", type=int, default=32)
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/results/experiments-13-18/seedlm_gpu_pilot"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = PUBLISHED[0 if args.bits == 3 else 1]
    encoder = GPUPaperEncoder(config, Path("build/paper-seedlm.so"),
                              args.seed_tile, args.block_tile)
    original = Weights("data/llama-2-7b-hf").get(0, "q_proj").ravel()
    blocks = original[:args.blocks*config.block].reshape(-1, config.block)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    gpu = encoder.fields(blocks)
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter()-started
    started = time.perf_counter()
    cpu = encoder.cpu.fields(blocks, exhaustive=True)
    cpu_seconds = time.perf_counter()-started
    gpu_decoded = decode(config, pack(config, *gpu), len(blocks))
    cpu_decoded = decode(config, pack(config, *cpu), len(blocks))
    gpu_error = np.square(blocks-gpu_decoded.astype(np.float32), dtype=np.float64).sum(1)
    cpu_error = np.square(blocks-cpu_decoded.astype(np.float32), dtype=np.float64).sum(1)
    excess = gpu_error-cpu_error
    rows = [dict(block=i, gpu_seed=int(gpu[0][i]), cpu_seed=int(cpu[0][i]),
                 gpu_error=float(gpu_error[i]), cpu_error=float(cpu_error[i]),
                 excess_error=float(excess[i]),
                 fields_equal=bool(all(np.array_equal(g[i], c[i]) for g, c in zip(gpu, cpu))))
            for i in range(len(blocks))]
    with (args.out / f"bits{args.bits}_parity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary = dict(bits=args.bits, blocks=len(blocks), seed_tile=args.seed_tile,
                   block_tile=args.block_tile, gpu_seconds=gpu_seconds,
                   cpu_seconds=cpu_seconds, gpu_blocks_per_second=len(blocks)/gpu_seconds,
                   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   field_mismatches=sum(not row["fields_equal"] for row in rows),
                   max_excess_error=float(excess.max()),
                   mean_excess_error=float(excess.mean()),
                   torch=torch.__version__, hip=torch.version.hip,
                   source="Llama-2-7B layer 0 q_proj contiguous blocks; all 65535 legal seeds")
    (args.out / f"bits{args.bits}_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
