"""Exact search and packed reference codec for signed-byte generators."""

import numpy as np

from experiments.generator_algebra import bitvector
from experiments.generator_algebra.bitvector import scalar_instruction


def exact_search(vectors, blocks, seed_batch=4096):
    """Minimize squared residual over every supplied vector and e in [-8, 7]."""
    blocks = np.asarray(blocks, dtype=np.float64)
    vectors = np.asarray(vectors)
    if (blocks.ndim != 2 or vectors.ndim != 2 or not len(blocks) or not len(vectors)
            or blocks.shape[1] != vectors.shape[1] or seed_batch < 1):
        raise ValueError("Expected nonempty matrices of equal width and a positive batch size")
    best_score = np.full(len(blocks), np.inf)
    best_seed = np.zeros(len(blocks), dtype=np.uint32)
    best_exponent = np.zeros(len(blocks), dtype=np.int8)
    columns = np.arange(len(blocks))
    for start in range(0, len(vectors), seed_batch):
        current = vectors[start:start + seed_batch].astype(np.float64)
        norm = np.einsum("iv,iv->i", current, current)[:, None]
        dot = current @ blocks.T
        optimum = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
        # For adjacent scales t and 2t, the boundary is their arithmetic midpoint.
        fraction, power = np.frexp(optimum)
        exponent = np.clip(power - 1 + (fraction > 0.75), -8, 7).astype(np.int8)
        exponent[optimum <= 0] = -8
        scale = np.ldexp(1.0, exponent)
        score = scale * (scale * norm - 2 * dot)
        row = score.argmin(axis=0)
        best = score[row, columns] < best_score
        best_score[best] = score[row, columns][best]
        best_seed[best] = (row + start)[best]
        best_exponent[best] = exponent[row, columns][best]
    reconstructed = np.ldexp(vectors[best_seed].astype(np.float64), best_exponent[:, None])
    errors = np.sum(np.square(blocks - reconstructed), axis=1)
    return errors, best_seed, best_exponent, reconstructed


def exact_program_search(program, blocks, seed_bits, seed_batch=4096):
    """Memory-bounded float64 exhaustive reference over a generator's full seed domain."""
    if not 1 <= seed_bits <= 24 or seed_batch < 1:
        raise ValueError("Invalid seed domain or batch size")
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.ndim != 2 or blocks.shape[1] != 8 or not len(blocks):
        raise ValueError("Expected nonempty eight-weight blocks")
    best_score = np.full(len(blocks), np.inf)
    best_seed = np.zeros(len(blocks), dtype=np.uint32)
    best_exponent = np.full(len(blocks), -8, dtype=np.int8)
    columns = np.arange(len(blocks))
    indices = np.arange(8, dtype=np.uint32)
    for start in range(0, 1 << seed_bits, seed_batch):
        stop = min(start + seed_batch, 1 << seed_bits)
        current = bitvector.evaluate(program, np.arange(start, stop, dtype=np.uint32)[:, None],
                                     indices).astype(np.float64)
        norm = np.einsum("iv,iv->i", current, current)[:, None]
        dot = current @ blocks.T
        optimum = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
        fraction, power = np.frexp(optimum)
        exponent = np.clip(power - 1 + (fraction > .75), -8, 7).astype(np.int8)
        exponent[optimum <= 0] = -8
        scale = np.ldexp(1.0, exponent)
        score = scale * (scale * norm - 2 * dot)
        row = score.argmin(axis=0)
        selected = score[row, columns]
        improve = selected < best_score
        best_score[improve] = selected[improve]
        best_seed[improve] = start + row[improve]
        best_exponent[improve] = exponent[row, columns][improve]
    reconstructed = np.ldexp(bitvector.evaluate(program, best_seed[:, None], indices).astype(np.float64),
                             best_exponent[:, None])
    errors = np.square(blocks - reconstructed).sum(axis=1)
    return errors, best_seed, best_exponent, reconstructed


def pack(seeds, exponents, seed_bits):
    """Pack consecutive little-endian fields with at most seven tail padding bits."""
    if not 1 <= seed_bits <= 24:
        raise ValueError("Seed field must be 1–24 bits")
    seeds, exponents = np.asarray(seeds), np.asarray(exponents)
    if (seeds.shape != exponents.shape or seeds.ndim != 1
            or not np.issubdtype(seeds.dtype, np.integer) or not np.issubdtype(exponents.dtype, np.integer)
            or np.any(seeds < 0)
            or np.any(seeds >= 1 << seed_bits) or np.any(exponents < -8) or np.any(exponents > 7)):
        raise ValueError("Invalid seed or exponent field")
    words = seeds.astype(np.uint32) | ((exponents.astype(np.uint32) & 15) << seed_bits)
    bits = (words[:, None] >> np.arange(seed_bits + 4, dtype=np.uint32)) & 1
    return np.packbits(bits.ravel(), bitorder="little").tobytes()


def decode(program, payload, seed_bits, width, *, count=None):
    """Independent scalar packed decoder, with no dependency on NumPy execution."""
    if not 1 <= seed_bits <= 24 or width < 1:
        raise ValueError("Invalid packed layout")
    bits = seed_bits + 4
    if count is None:
        if bits % 8:
            raise ValueError("Block count required for a non-byte-aligned format")
        count = len(payload) * 8 // bits
    padding = len(payload) * 8 - count * bits
    if count < 0 or not 0 <= padding < 8:
        raise ValueError("Invalid packed stream length")
    if padding and payload[-1] >> (8 - padding):
        raise ValueError("Nonzero padding in packed stream")
    result = []
    for block in range(count):
        byte, shift = divmod(block * bits, 8)
        stop = byte + (shift + bits + 7) // 8
        word = (int.from_bytes(payload[byte:stop], "little") >> shift) & ((1 << bits) - 1)
        seed, exponent = word & ((1 << seed_bits) - 1), word >> seed_bits
        if exponent >= 8:
            exponent -= 16
        row = []
        for index in range(width):
            values = {"s": seed, "i": index}
            for step, instruction in enumerate(program):
                args = [values[arg] if isinstance(arg, str) else arg for arg in instruction.args]
                values[f"r{step}"] = scalar_instruction(instruction.op, args)
            byte = values[f"r{len(program) - 1}"] % 256
            row.append((byte - 256 if byte >= 128 else byte) * 2.0 ** exponent)
        result.append(row)
    return result
