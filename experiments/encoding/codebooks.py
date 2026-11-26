"""Matched-rate vector and trellis baselines for the procedural weight study."""
from functools import lru_cache
from itertools import combinations, permutations, product

import numpy as np

from experiments.encoding.discrete_oracle import optimal_power
from experiments.generator_algebra.dictionary import hash32


def computed_vectors(seeds, width, salt=0):
    """Approximate Gaussian via four byte lanes of a 32-bit hash."""
    seeds = np.asarray(seeds, dtype=np.uint32)
    words = hash32((seeds[:, None] * np.uint32(32) + np.arange(width, dtype=np.uint32)) ^
                   np.uint32(salt))
    values = sum(((words >> shift) & 255).astype(np.int32) for shift in (0, 8, 16, 24))
    return (values - np.int32(510)).astype(np.float32) / np.float32(147.800537109375)


def dot_norm(vectors, blocks):
    return np.square(vectors).sum(axis=1), vectors @ blocks.T


def nearest(vectors, blocks, norm=None):
    if norm is None:
        norm = np.square(vectors).sum(axis=1)
    return np.argmin(norm[:, None] - 2 * (vectors @ blocks.T), axis=0)


def dyadic_search(vectors, blocks, batch=32768, signed=False):
    """Exhaustive vector and exponent search. Optional independently stored signs."""
    targets = np.abs(blocks) if signed else blocks
    best = np.full(len(blocks), np.inf)
    ids = np.zeros(len(blocks), dtype=np.int64)
    exponents = np.zeros(len(blocks), dtype=np.int8)
    cols = np.arange(len(blocks))
    for start in range(0, len(vectors), batch):
        norm, dot = dot_norm(vectors[start:start + batch], targets)
        cost, exp = optimal_power(norm[:, None], dot, (-8, 7))
        winner = cost.argmin(axis=0)
        values = cost[winner, cols]
        improved = values < best
        best[improved] = values[improved]
        ids[improved] = start + winner[improved]
        exponents[improved] = exp[winner, cols][improved]
    return ids, exponents


def direct(vectors, blocks):
    ids, exponents = dyadic_search(vectors, blocks)
    return vectors[ids] * np.exp2(exponents.astype(np.float32))[:, None]


def additive(first, second, blocks, sweeps=2, starts=4):
    """Use several initial seeds and alternately solve both one-seed problems."""
    energy = np.square(blocks).sum(axis=1, dtype=np.float64)
    best = energy.copy()
    # The all-zero pair is reserved as an always available output.
    result = np.zeros_like(blocks)
    norm_first = np.square(first).sum(axis=1)
    norm_second = np.square(second).sum(axis=1)
    for exp in range(-8, 8):
        target = blocks / np.float32(2.0 ** exp)
        cost = norm_first[:, None] - 2 * (first @ target.T)
        left = np.argpartition(cost, starts - 1, axis=0)[:starts].ravel()
        expanded = np.tile(target, (starts, 1))
        for _ in range(sweeps):
            right = nearest(second, expanded - first[left], norm_second)
            left = nearest(first, expanded - second[right], norm_first)
        candidate = ((first[left] + second[right]) * np.float32(2.0 ** exp)).reshape(
            starts, *blocks.shape)
        errors = np.square(candidate - blocks).sum(axis=2)
        cols = np.arange(len(blocks))
        winner = errors.argmin(axis=0)
        values = errors[winner, cols]
        improved = values < best
        result[improved] = candidate[winner, cols][improved]
        best[improved] = values[improved]
    return result


def permutation_vectors():
    """Sixteen generated magnitude templates with lexicographic permutation ranks."""
    templates = np.sort(np.abs(computed_vectors(np.arange(16), 8, 0xE38533)), axis=1)
    orders = np.array(list(permutations(range(8))), dtype=np.uint8)
    return templates[:, orders].reshape(-1, 8)


def signed_permutation(vectors, blocks):
    ids, exponents = dyadic_search(vectors, blocks, signed=True)
    signs = np.where(np.signbit(blocks), np.float32(-1), np.float32(1))
    return signs * vectors[ids] * np.exp2(exponents.astype(np.float32))[:, None]


def lattice_vectors():
    """E8P-style shaped E8 code with a deterministic choice of padding shell."""
    magnitudes = np.array(list(product((0.5, 1.5, 2.5, 3.5), repeat=8)), dtype=np.float32)
    core = magnitudes[np.square(magnitudes).sum(axis=1) <= 10]
    padding = np.full((29, 8), 0.5, dtype=np.float32)
    for row, axes in enumerate(list(combinations(range(8), 5))[:len(padding)]):
        padding[row, list(axes)] = 1.5
    shapes = np.vstack((core, padding))
    signs = 1 - 2 * (np.arange(128)[:, None] >> np.arange(7) & 1)
    vectors = shapes[:, None, :] * np.column_stack((signs, np.ones(128, dtype=int)))
    odd = vectors.sum(axis=-1) % 2 != 0
    vectors[:, :, -1] *= 1 - 2 * odd
    return np.stack((vectors - 0.25, vectors + 0.25)).reshape(-1, 8).astype(np.float32)


def lattice_quantize(blocks, codebook, scale):
    ids = nearest(codebook, blocks / scale)
    return codebook[ids] * scale


@lru_cache(maxsize=16)
def one_mad(bits):
    # Scalar distribution defined by the QTIP 1MAD reconstruction map.
    state = np.arange(1 << bits, dtype=np.uint32)
    words = state * np.uint32(34038481) + np.uint32(76625530)
    byte_sum = sum(((words >> shift) & 255).astype(np.int32) for shift in (0, 8, 16, 24))
    values = (byte_sum.astype(np.float32) - 510) / np.float32(147.800537109375)
    values.setflags(write=False)
    return values


def trellis_row(row, width, state_bits=16):
    """Exact open-boundary Viterbi with a known zero initial state."""
    lookup = one_mad(state_bits)
    states = 1 << state_bits
    symbols = 1 << width
    groups = states // symbols
    costs = np.full(states, np.inf, dtype=np.float32)
    costs[0] = 0
    history = np.empty((len(row), groups), dtype=np.uint8)
    columns = np.arange(groups)
    for index, value in enumerate(row):
        predecessors = costs.reshape(symbols, groups)
        winners = predecessors.argmin(axis=0)
        history[index] = winners
        costs = np.repeat(predecessors[winners, columns], symbols) + np.square(lookup - value)
    state = int(costs.argmin())
    path = np.empty(len(row), dtype=np.uint8)
    for index in range(len(row) - 1, -1, -1):
        path[index] = state & (symbols - 1)
        group = state >> width
        state = int(history[index, group]) * groups + group
    state = 0
    decoded = np.empty_like(row, dtype=np.float32)
    for index, symbol in enumerate(path):
        state = ((state << width) | int(symbol)) & (states - 1)
        decoded[index] = lookup[state]
    return decoded


def hadamard_rows(rows, inverse=False):
    """Orthogonal transform within each fixed 64-column group."""
    transformed = np.asarray(rows, dtype=np.float32).reshape(-1, 64).copy()
    signs = 1 - 2 * (hash32(np.arange(64, dtype=np.uint32) ^ np.uint32(0xDB2B99))
                     & 1).astype(np.float32)
    if not inverse:
        transformed *= signs
    for step in (1, 2, 4, 8, 16, 32):
        groups = transformed.reshape(-1, 2, step)
        first = groups[:, 0].copy()
        second = groups[:, 1].copy()
        groups[:, 0] = first + second
        groups[:, 1] = first - second
    transformed *= np.float32(0.125)
    if inverse:
        transformed *= signs
    return transformed.reshape(rows.shape)
