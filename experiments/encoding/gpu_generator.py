"""Tiled exhaustive ROCm/PyTorch search for the frozen eight-weight generator.

All seeds are visited in ascending order. FP64 dot products and scale scoring
avoid a reduced-precision shortlist; exact fields still require CPU auditing
because GPU and NumPy reduction orders can differ at numerical ties.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.encoding import synthesized
from experiments.generator_algebra import bitvector


class GPUEncoder:
    def __init__(self, program, seed_bits=24, *, seed_tile=16384, target_tile=256,
                 enumeration_tile=65536, cache_dir=None, score_backend="eager"):
        if not 1 <= seed_bits <= 24 or min(seed_tile, target_tile, enumeration_tile) < 1:
            raise ValueError("Invalid GPU encoder tile or seed field")
        if score_backend not in ("eager", "compiled"):
            raise ValueError("Invalid score backend")
        import torch
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("GPUEncoder requires a working ROCm PyTorch device")
        self.torch = torch
        self.program = program
        self.seed_bits = seed_bits
        self.seed_tile = seed_tile
        self.target_tile = target_tile
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.policy = dict(seed_bits=seed_bits, seed_tile=seed_tile, target_tile=target_tile,
                           dtype="float64", exponent="frexp midpoint", tie="lowest seed then exponent",
                           score_backend=score_backend)
        self.score_backend = score_backend
        tick = time.perf_counter()
        count = 1 << seed_bits
        self.codebook = torch.empty((count, 8), dtype=torch.float64, device="cuda")
        indices = np.arange(8, dtype=np.uint32)
        for start in range(0, count, enumeration_tile):
            stop = min(count, start + enumeration_tile)
            seeds = np.arange(start, stop, dtype=np.uint32)
            vectors = bitvector.evaluate(program, seeds[:, None], indices)
            self.codebook[start:stop] = torch.from_numpy(vectors.astype(np.float64)).to("cuda")
        self.norms = (self.codebook * self.codebook).sum(dim=1)
        torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter() - tick
        def score_min(dot, norm, columns):
            optimum = torch.where(norm > 0, dot / torch.clamp(norm, min=1), 0)
            fraction, power = torch.frexp(optimum)
            exponent = torch.clamp(power - 1 + (fraction > .75).to(power.dtype), -8, 7)
            exponent = torch.where(optimum > 0, exponent, -8)
            scale = torch.ldexp(torch.ones_like(dot), exponent)
            score = scale * (scale * norm - 2 * dot)
            local_score, local_seed = score.min(dim=0)
            selected_exponent = exponent[local_seed, columns].to(torch.int8)
            return local_score, local_seed, selected_exponent
        self._score_min = (torch.compile(score_min, fullgraph=True)
                           if score_backend == "compiled" else score_min)
        self.jit_seconds = 0.0
        if score_backend == "compiled":
            tick = time.perf_counter()
            rows = min(seed_tile, count)
            dummy = torch.zeros((rows, target_tile), dtype=torch.float64, device="cuda")
            columns = torch.arange(target_tile, device="cuda")
            self._score_min(dummy, self.norms[:rows, None], columns)
            torch.cuda.synchronize()
            self.jit_seconds = time.perf_counter() - tick
            del dummy
        self.transfer_seconds = 0.0
        self.kernel_seconds = 0.0
        self.blocks_encoded = 0

    def fields(self, blocks):
        torch = self.torch
        blocks = np.asarray(blocks, dtype=np.float64)
        if blocks.ndim != 2 or blocks.shape[1] != 8:
            raise ValueError("Expected eight-weight blocks")
        seeds = np.empty(len(blocks), dtype=np.uint32)
        exponents = np.empty(len(blocks), dtype=np.int8)
        for target_start in range(0, len(blocks), self.target_tile):
            target_stop = min(len(blocks), target_start + self.target_tile)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            if self.score_backend == "compiled":
                padded = np.zeros((self.target_tile, 8), dtype=np.float64)
                padded[:target_stop - target_start] = blocks[target_start:target_stop]
                targets = torch.from_numpy(padded).to("cuda")
            else:
                targets = torch.from_numpy(blocks[target_start:target_stop]).to("cuda")
            torch.cuda.synchronize()
            self.transfer_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            best_score = torch.full((len(targets),), float("inf"), device="cuda", dtype=torch.float64)
            best_seed = torch.zeros((len(targets),), device="cuda", dtype=torch.int64)
            best_exponent = torch.full((len(targets),), -8, device="cuda", dtype=torch.int8)
            columns = torch.arange(len(targets), device="cuda")
            for seed_start in range(0, len(self.codebook), self.seed_tile):
                seed_stop = min(len(self.codebook), seed_start + self.seed_tile)
                dot = self.codebook[seed_start:seed_stop] @ targets.T
                norm = self.norms[seed_start:seed_stop, None]
                local_score, local_seed, selected_exponent = self._score_min(dot, norm, columns)
                improve = local_score < best_score
                best_score = torch.where(improve, local_score, best_score)
                best_seed = torch.where(improve, local_seed + seed_start, best_seed)
                best_exponent = torch.where(improve, selected_exponent, best_exponent)
            torch.cuda.synchronize()
            self.kernel_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            seeds[target_start:target_stop] = best_seed.cpu().numpy()[:target_stop - target_start]
            exponents[target_start:target_stop] = best_exponent.cpu().numpy()[:target_stop - target_start]
            self.transfer_seconds += time.perf_counter() - tick
        self.blocks_encoded += len(blocks)
        return seeds, exponents

    def encode(self, blocks):
        return synthesized.pack(*self.fields(blocks), self.seed_bits)

    def encode_matrix(self, matrix):
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[1] % 8:
            raise ValueError("Expected matrix rows containing whole eight-weight blocks")
        blocks = matrix.reshape(-1, 8)
        seeds = np.empty(len(blocks), dtype=np.uint32)
        exponents = np.empty(len(blocks), dtype=np.int8)
        identity = hashlib.sha256()
        identity.update(np.ascontiguousarray(matrix).tobytes())
        identity.update(str(matrix.shape).encode())
        identity.update(bitvector.expression(self.program).encode())
        identity.update(json.dumps(self.policy, sort_keys=True).encode())
        key = identity.hexdigest()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        for start in range(0, len(blocks), 32768):
            stop = min(start + 32768, len(blocks))
            path = self.cache_dir / f"{key}-{start}-{stop}.npz" if self.cache_dir else None
            if path is not None and path.is_file():
                with np.load(path) as saved:
                    if saved["identity"].item() != key or len(saved["seeds"]) != stop - start:
                        raise ValueError(f"Invalid cached fields: {path}")
                    seeds[start:stop], exponents[start:stop] = saved["seeds"], saved["exponents"]
            else:
                seeds[start:stop], exponents[start:stop] = self.fields(blocks[start:stop])
                if path is not None:
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez(temporary, identity=key, seeds=seeds[start:stop], exponents=exponents[start:stop])
                    temporary.replace(path)
            elapsed = time.perf_counter() - started
            print(f"encoded {stop:,}/{len(blocks):,} blocks in {elapsed:.1f}s "
                  f"({stop / max(elapsed, 1e-9):,.1f} blocks/s)", flush=True)
        payload = synthesized.pack(seeds, exponents, self.seed_bits)
        vectors = bitvector.evaluate(self.program, seeds[:, None], np.arange(8, dtype=np.uint32))
        restored = np.ldexp(vectors.astype(np.float32), exponents[:, None]).reshape(matrix.shape)
        count = min(len(blocks), 16)
        prefix = payload[:(count * (self.seed_bits + 4) + 7) // 8]
        np.testing.assert_array_equal(synthesized.decode(self.program, prefix, self.seed_bits, 8, count=count),
                                      restored.reshape(-1, 8)[:count])
        return restored, payload
