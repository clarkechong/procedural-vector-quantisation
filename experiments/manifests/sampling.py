"""Shared disjoint-row sampling for the SmolLM2 codec studies."""

import numpy as np

from experiments.manifests.weights import PROJECTIONS


SPAN = 192  # Common multiple of the tested block lengths.


def samples(weights, layers, spans, rng, *, row_manifest=None):
    """Hold out disjoint original rows while retaining identical spans across codecs."""
    datasets = {"screen": {}, "holdout": {}}
    for layer in layers:
        for projection in PROJECTIONS:
            matrix = weights.get(layer, projection)
            if matrix.shape[0] < 2 * spans or matrix.shape[1] < SPAN:
                raise ValueError(f"Not enough rows or columns for paired 192-weight samples at {layer}/{projection}")
            sampled_rows = rng.choice(matrix.shape[0], 2 * spans, replace=False)
            if row_manifest is not None:
                row_manifest[layer, projection] = sampled_rows.tolist()
            starts = rng.integers(matrix.shape[1] // SPAN, size=2 * spans) * SPAN
            slices = np.stack([matrix[row, start:start + SPAN] for row, start in zip(sampled_rows, starts)])
            mean = matrix.mean(dtype=np.float64)
            std = matrix.std(dtype=np.float64)
            for split, original in zip(datasets, np.split(slices, 2)):
                indices = rng.choice(matrix.size, original.size, replace=False)
                shuffled = matrix.ravel()[indices].reshape(original.shape)
                noise = rng.standard_normal(original.shape, dtype=np.float32)
                gaussian = (noise * std + mean).astype(np.float32)
                datasets[split][layer, projection] = (original, shuffled, gaussian)
    return datasets
