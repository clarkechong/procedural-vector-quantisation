"""Rate-distortion measurements with fixed-width payload accounting."""

import math

import numpy as np

from experiments.generator_algebra.dictionary import parity, random_signs


def fp16(values):
    return values.astype(np.float16).astype(np.float32)


def sse(original, reconstructed):
    return np.sum(np.square(original - reconstructed, dtype=np.float64), axis=-1)


def pursuit(blocks, atoms, steps):
    # Update inner products through the Gram matrix. FP16 coefficient rounding
    # is applied before updating the residual and recording its SSE.
    gram = (atoms @ atoms.T).astype(np.float64)
    diagonal = np.diag(gram)
    correlations = (blocks @ atoms.T).astype(np.float64)
    energy = sse(blocks, 0)
    rows = np.arange(len(blocks))
    for count in range(1, steps + 1):
        coefficients = fp16(correlations / diagonal)
        gain = coefficients * (2 * correlations - coefficients * diagonal)
        index = np.argmax(gain, axis=1)
        coefficient = coefficients[rows, index]
        energy -= gain[rows, index]
        correlations -= coefficient[:, None] * gram[index]
        yield count, np.maximum(energy, 0)


def seeded_basis(blocks, seeds, counts):
    # A fixed sign mask per seed times Walsh columns. On power-of-two block
    # areas the generated vectors are orthogonal, so fitting uses projections.
    length = blocks.shape[1]
    padded = 1 << (length - 1).bit_length()
    indices = np.arange(padded, dtype=np.uint32)
    count = max(counts)
    walsh = parity(np.arange(count, dtype=np.uint32)[:, None] & indices) / math.sqrt(padded)
    walsh = walsh[:, :length].copy()
    orthogonal = padded == length
    inverses = {} if orthogonal else {
        size: np.linalg.pinv(walsh[:size] @ walsh[:size].T) for size in counts
    }
    best = {size: np.zeros(len(blocks), dtype=np.float64) for size in counts}
    energy = sse(blocks, 0)
    seed_batch = max(1, min(seeds, 2_000_000 // (length * count)))
    for start in range(0, seeds, seed_batch):
        basis = np.concatenate([
            walsh * random_signs(indices[:length], seed)
            for seed in range(start, min(start + seed_batch, seeds))
        ])
        correlations = (blocks @ basis.T).reshape(len(blocks), -1, count)
        if orthogonal:
            rounded = fp16(correlations)
            gains = np.cumsum(rounded * (2 * correlations - rounded), axis=2)
            for size in counts:
                np.maximum(best[size], gains[:, :, size - 1].max(axis=1), out=best[size])
        else:
            gram = walsh @ walsh.T
            for size in counts:
                dots = correlations[:, :, :size]
                rounded = fp16(dots @ inverses[size])
                gains = np.sum(rounded * (2 * dots - rounded @ gram[:size, :size]), axis=2)
                np.maximum(best[size], gains.max(axis=1), out=best[size])
    for size in counts:
        yield size, np.maximum(energy - best[size], 0)


def quantize(values, depth, nonuniform):
    levels = 2 ** depth
    if nonuniform:
        ordered = np.sort(values, axis=1)
        offsets = ((np.arange(levels) + 0.5) * values.shape[1] / levels).astype(int)
        centers = ordered[:, offsets]
        for _ in range(8):
            boundaries = (centers[:, 1:] + centers[:, :-1]) / 2
            for row, samples in enumerate(values):
                labels = np.searchsorted(boundaries[row], samples)
                totals = np.bincount(labels, weights=samples, minlength=levels)
                counts = np.bincount(labels, minlength=levels)
                np.divide(totals, counts, out=centers[row], where=counts > 0)
        centers = fp16(centers)
        boundaries = (centers[:, 1:] + centers[:, :-1]) / 2
        decoded = np.stack([
            centers[row, np.searchsorted(boundaries[row], samples)]
            for row, samples in enumerate(values)
        ])
        return depth * values.shape[1] + 16 * levels, sse(values, decoded)
    offset = fp16(values.min(axis=1, keepdims=True))
    scale = fp16((values.max(axis=1, keepdims=True) - offset) / (levels - 1))
    index = np.divide(values - offset, scale, out=np.zeros_like(values), where=scale > 0)
    decoded = offset + np.clip(np.rint(index), 0, levels - 1) * scale
    return depth * values.shape[1] + 32, sse(values, decoded)


def low_rank(tiles, ranks):
    height, width = tiles.shape[-2:]
    if min(height, width) > 64:
        # One fixed randomized range finder, shared across all datasets.
        sketch = np.random.default_rng(2026).standard_normal((width, max(ranks) + 8), dtype=np.float32)
        sample = tiles @ sketch
        for _ in range(2):
            q, _ = np.linalg.qr(sample)
            sample = tiles @ (tiles.transpose(0, 2, 1) @ q)
        q, _ = np.linalg.qr(sample)
        u, singular, vh = np.linalg.svd(q.transpose(0, 2, 1) @ tiles, full_matrices=False)
        u = q @ u
    else:
        u, singular, vh = np.linalg.svd(tiles, full_matrices=False)
    root = np.sqrt(singular[:, :max(ranks)])
    left = fp16(u[:, :, :max(ranks)] * root[:, None, :])
    right = fp16(root[:, :, None] * vh[:, :max(ranks)])
    for rank in ranks:
        estimate = left[:, :, :rank] @ right[:, :rank]
        yield 16 * rank * (height + width), sse(tiles.reshape(len(tiles), -1),
                                                 estimate.reshape(len(tiles), -1))


def curves(tiles, atoms, families, seeds, maximum_rate, max_atoms):
    blocks = tiles.reshape(len(tiles), -1)
    area = blocks.shape[1]
    budget = maximum_rate * area
    steps = min(max_atoms, int(budget // 24))
    for method, selected in (
        ("procedural", atoms),
        ("coordinate_only", atoms[families != "random"]),
        ("random_atoms", atoms[families == "random"]),
    ):
        for count, error in pursuit(blocks, selected, steps):
            yield method, str(count), 24 * count, error
    counts = [count for count in (1, 2, 4, 8, 16) if count <= area and 8 + 16 * count <= budget]
    if counts:
        for count, error in seeded_basis(blocks, seeds, counts):
            yield "seeded_basis", str(count), 8 + 16 * count, error

    # Choose the k largest magnitudes, then store each index and FP16 value.
    sparse_cost = 16 + (area - 1).bit_length()
    counts = sorted({int(rate * area // sparse_cost) for rate in (0.25, 0.5, 1, 2, 3, 4, 6, 8)
                     if rate <= maximum_rate} - {0})
    if counts:
        ordered = np.take_along_axis(blocks, np.argsort(-np.abs(blocks), axis=1), axis=1)
        rounded = fp16(ordered[:, :max(counts)])
        gain = np.cumsum(rounded * (2 * ordered[:, :max(counts)] - rounded), axis=1,
                         dtype=np.float64)
        energy = sse(blocks, 0)
        for count in counts:
            yield "sparse", str(count), sparse_cost * count, np.maximum(energy - gain[:, count - 1], 0)

    # Quantizers use one codebook or offset/scale pair per block.
    for depth in (1, 2, 3, 4, 6, 8):
        for method, nonuniform in (("uniform", False), ("nonuniform", True)):
            overhead = 16 * (2 ** depth) if nonuniform else 32
            if area * depth + overhead <= budget:
                bits, error = quantize(blocks, depth, nonuniform)
                yield method, str(depth), bits, error

    factor_bits = 16 * sum(tiles.shape[-2:])
    ranks = [rank for rank in (1, 2, 4, 8, 16)
             if rank <= min(tiles.shape[-2:]) and rank * factor_bits <= budget]
    if ranks:
        for rank, (bits, error) in zip(ranks, low_rank(tiles, ranks)):
            yield "svd", str(rank), bits, error
