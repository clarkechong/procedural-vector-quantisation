"""Exact weighted Viterbi for a fixed map and scale."""

import numpy as np


def encode(row, weights, state_bits, stride, values):
    target = np.asarray(row, dtype=np.float64)
    metric = np.asarray(weights, dtype=np.float64)
    lookup = np.asarray(values, dtype=np.float64)
    if target.ndim != 1 or target.shape != metric.shape or len(target) == 0:
        raise ValueError("target and metric must be equal nonempty vectors")
    if not np.isfinite(metric).all() or (metric < 0).any() or not np.isfinite(target).all():
        raise ValueError("metric must be finite and nonnegative")
    if not 0 < stride <= state_bits <= 16 or lookup.shape != (1 << state_bits,):
        raise ValueError("invalid trellis geometry")
    symbols = 1 << stride
    groups = len(lookup) // symbols
    costs = metric[0] * np.square(lookup - target[0])
    history = np.empty((len(target) - 1, groups), dtype=np.uint16)
    for i in range(1, len(target)):
        previous = costs.reshape(groups, symbols)
        selected = previous.argmin(axis=1)
        history[i-1] = selected
        costs = np.tile(previous[np.arange(groups), selected], symbols) + metric[i] * np.square(lookup - target[i])
    state = int(costs.argmin())
    objective = float(costs[state])
    path = np.empty(len(target), dtype=np.int32)
    path[-1] = state
    for i in range(len(target)-2, -1, -1):
        group = state & (groups-1)
        state = (group << stride) | int(history[i, group])
        path[i] = state
    payload = int(path[0])
    for i in range(1, len(path)):
        payload |= int(path[i] >> (state_bits-stride)) << (state_bits + stride*(i-1))
    return lookup[path], payload, objective


def refit_scale(target, metric, assigned_unscaled, old_scale):
    numerator = np.sum(metric * target * assigned_unscaled)
    denominator = np.sum(metric * np.square(assigned_unscaled))
    return max(float(numerator / denominator), 1e-12) if denominator > 0 else old_scale


def exact_circular_small(row, weights, state_bits, stride, values):
    """Exhaustive tail-biting oracle for tiny fixtures (at most 16 payload bits)."""
    from experiments.synthesis.trellis import decode_bits
    total = state_bits + stride*(len(row)-1)
    if total > 16:
        raise ValueError("small exhaustive oracle only")
    groups = (1 << (state_bits-stride)) - 1
    best = (float("inf"), None, None)
    for payload in range(1 << total):
        first = payload & ((1 << state_bits)-1)
        last = (payload >> (stride*(len(row)-1))) & ((1 << state_bits)-1)
        if (first >> stride) != (last & groups):
            continue
        decoded = decode_bits(payload, len(row), state_bits, stride, values)
        cost = float(np.sum(np.asarray(weights)*(np.asarray(row)-decoded)**2))
        if cost < best[0]:
            best = (cost, payload, decoded)
    return best[2], best[1], best[0]
