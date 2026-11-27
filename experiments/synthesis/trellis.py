"""Exact finite-state encoding for overlapping latent-bit windows.

The first output consumes ``state_bits`` payload bits; each subsequent output
introduces ``stride`` more. Windows are low-bit first, with an open boundary.
"""

import numpy as np


def signed_byte(values):
    """Interpret the low eight bits as two's-complement signed integers."""
    return (np.asarray(values, dtype=np.int64) & 255).astype(np.uint8).view(np.int8)


def linear_map(state_bits, multiplier=120):
    states = np.arange(1 << state_bits, dtype=np.int64)
    return signed_byte(states * multiplier).astype(np.float64)


def decode_bits(bits, length, state_bits, stride, values):
    """Decode an integer bitstream; useful as an independent exhaustive oracle."""
    if not 0 < stride <= state_bits or length < 1:
        raise ValueError("Invalid window geometry")
    if len(values) != 1 << state_bits:
        raise ValueError("Mapping size does not match the state width")
    mask = (1 << state_bits) - 1
    return np.asarray([values[(bits >> (stride * i)) & mask]
                       for i in range(length)], dtype=np.float64)


def encode(row, state_bits, stride, values):
    """Find the minimum squared-error bitstream with Viterbi dynamic programming.

    Returns the reconstructed row, payload integer, and squared error. A
    traceback costs (length-1)*2**(state_bits-stride) bytes when stride<=8.
    """
    target = np.asarray(row, dtype=np.float64)
    lookup = np.asarray(values, dtype=np.float64)
    if target.ndim != 1 or len(target) < 1:
        raise ValueError("Expected a nonempty one-dimensional row")
    if not 0 < stride <= state_bits or stride > 8 or state_bits > 20:
        raise ValueError("Unsupported window geometry")
    states = 1 << state_bits
    if lookup.shape != (states,) or not np.isfinite(lookup).all():
        raise ValueError("Mapping must contain one finite value per state")
    symbols = 1 << stride
    groups = states // symbols
    costs = np.square(lookup - target[0])
    history = np.empty((len(target) - 1, groups), dtype=np.uint8)
    columns = np.arange(groups)
    for i in range(1, len(target)):
        predecessors = costs.reshape(groups, symbols)
        best = predecessors.argmin(axis=1)
        history[i - 1] = best
        costs = np.tile(predecessors[columns, best], symbols) + np.square(lookup - target[i])
    state = int(costs.argmin())
    error = float(costs[state])
    path = np.empty(len(target), dtype=np.int32)
    path[-1] = state
    for i in range(len(target) - 2, -1, -1):
        group = state & (groups - 1)
        state = (group << stride) | int(history[i, group])
        path[i] = state
    payload = int(path[0])
    for i in range(1, len(path)):
        payload |= int(path[i] >> (state_bits - stride)) << (state_bits + stride * (i - 1))
    reconstructed = lookup[path]
    return reconstructed, payload, error


def dependency_graph(program, seed_bits, width):
    """Exact Boolean influence graph by flipping every seed bit in every seed."""
    from experiments.generator_algebra import bitvector

    if seed_bits > 16:
        raise ValueError("Exhaustive graph analysis is limited to 16 seed bits")
    book = bitvector.codebook(program, seed_bits, width)
    seeds = np.arange(1 << seed_bits)
    return np.asarray([np.any(book != book[seeds ^ (1 << bit)], axis=0)
                       for bit in range(seed_bits)], dtype=bool).T


def window_dependency(state_bits, stride, length, values):
    """Exact influence of each payload bit, including mapping cancellations."""
    values = np.asarray(values)
    active = np.asarray([np.any(values != values[np.arange(len(values)) ^ (1 << bit)])
                         for bit in range(state_bits)])
    total = state_bits + stride * (length - 1)
    graph = np.zeros((length, total), dtype=bool)
    for i in range(length):
        graph[i, stride * i:stride * i + state_bits] = active
    return graph
