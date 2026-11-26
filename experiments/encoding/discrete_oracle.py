"""Exact SeedLM encoder on small samples using projection lower bounds."""
from itertools import product

import numpy as np

from experiments.encoding.seedlm import basis, seed_ids


def integer_codes(config):
    limit = 1 << (config.coefficient_bits - 1)
    return np.array(list(product(range(-limit, limit), repeat=config.latent)), dtype=np.float64)


def optimal_power(norm, dot, limits=None):
    """Minimize ||w - 2**e v||² for every direction v."""
    gain = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
    positive = gain > 0
    low = np.frexp(np.where(positive, gain, 1))[1] - 1
    high = low + 1
    if limits is not None:
        low = np.clip(np.where(positive, low, limits[0]), *limits)
        high = np.clip(np.where(positive, high, limits[0]), *limits)
    best_exponents = low.copy()
    low_scale, high_scale = np.exp2(low), np.exp2(high)
    best = low_scale * (low_scale * norm - 2 * dot)
    alternative = high_scale * (high_scale * norm - 2 * dot)
    improved = alternative < best
    best[improved] = alternative[improved]
    best_exponents[improved] = high[improved]
    if limits is None:
        best[~positive] = 0
    return best, best_exponents


def exact(config, blocks, incumbent):
    """Search every seed that can beat the incumbent. Return codewords and bounds."""
    blocks = np.asarray(blocks, dtype=np.float64)
    seeds = seed_ids(config)
    bases = basis(config, seeds).astype(np.float64)
    inverses = np.linalg.pinv(bases, rcond=1e-13)
    codes = integer_codes(config)
    output = []
    initial_errors, winner_seeds, winner_codes, winner_exponents = incumbent
    for index, block in enumerate(blocks):
        dots = bases.transpose(0, 2, 1) @ block
        continuous = (inverses @ block[..., None])[..., 0]
        energy = block @ block
        bounds = np.square((bases @ continuous[..., None])[..., 0] - block).sum(axis=1)
        best_seed, best_code, best_exponent = winner_seeds[index], winner_codes[index], winner_exponents[index]
        reconstruction = basis(config, [best_seed])[0].astype(np.float64) @ best_code
        best = np.square(reconstruction * np.exp2(float(best_exponent)) - block).sum()
        np.testing.assert_allclose(best, initial_errors[index], rtol=5e-5, atol=2e-8)
        uncapped, free_scale = best, best
        visited = 0
        # Pruning at the largest remaining incumbent certifies all three oracles.
        for candidate in np.argsort(bounds):
            if bounds[candidate] > best + 1e-12 * energy:
                break
            visited += 1
            vectors = codes @ bases[candidate].T
            norm = np.einsum("ij,ij->i", vectors, vectors)
            dot = codes @ dots[candidate]
            positive = np.maximum(dot, 0)
            gain = np.divide(positive ** 2, norm, out=np.zeros_like(norm), where=norm > 0)
            free_scale = min(free_scale, energy - gain.max())
            cost, _ = optimal_power(norm, dot)
            uncapped = min(uncapped, energy + cost.min())
            cost, exponents = optimal_power(norm, dot, (-8, 7))
            choice = cost.argmin()
            if energy + cost[choice] < best:
                best = energy + cost[choice]
                best_seed, best_code, best_exponent = seeds[candidate], codes[choice], exponents[choice]
        output.append((best, best_seed, best_code, best_exponent, bounds.min(), uncapped,
                       free_scale, visited))
    return output
