"""Fixed coordinate generators and reproducible counter-based pseudorandom signs."""

import numpy as np


def hash32(word):
    word = (word ^ (word >> 16)) * np.uint32(0x7FEB352D)
    word = (word ^ (word >> 15)) * np.uint32(0x846CA68B)
    return word ^ (word >> 16)


def random_signs(index, seed):
    word = np.asarray(index, dtype=np.uint32) ^ np.uint32((0x9E3779B9 * (seed + 1)) & 0xFFFFFFFF)
    return (hash32(word) & 1).astype(np.float32) * 2 - 1


def parity(word):
    return 1 - 2 * (np.bitwise_count(word) % 2).astype(np.float32)


def dictionary(shape, seeds):
    height, width = shape
    i = np.arange(height, dtype=np.uint32)[:, None]
    j = np.arange(width, dtype=np.uint32)[None, :]
    atoms, families = [], []

    def add(family, atom):
        values = np.broadcast_to(atom, shape).ravel().astype(np.float32)
        norm = np.linalg.norm(values)
        if norm:
            atoms.append(values / norm)
            families.append(family)

    add("constant", 1)
    add("affine", (i.astype(np.float32) - (height - 1) / 2) / max(1, height))
    add("affine", (j.astype(np.float32) - (width - 1) / 2) / max(1, width))
    if height > 1:
        add("bitwise", (i ^ j).astype(np.float32) / max(height, width) - 0.5)
        add("walsh", parity(i & j))
        add("walsh", parity((i ^ 1) & j))
        add("walsh", parity(i & (j ^ 1)))
    for period in (2, 4, 8):
        for phase in range(period):
            add("periodic", ((i + j) % period == phase).astype(np.float32) - 1 / period)
    for axis in (i, j):
        size = axis.size
        for groups in (2, 4, 8):
            if size >= groups:
                for group in range(groups):
                    add("block", (axis * groups // size == group).astype(np.float32) - 1 / groups)
    for seed in range(seeds):
        add("random", random_signs(i * width + j, seed))
    if len(atoms) > 256:
        raise ValueError("The dictionary must fit in an 8-bit atom identifier")
    return np.stack(atoms), np.array(families)


def planted(shape, rng):
    """Independent combinations from known families, with held-out coefficients."""
    height, width = shape
    i, j = np.indices(shape, dtype=np.uint32)
    phase = rng.integers(4)
    periodic = 2 * (((i + j) % 4 == phase).astype(np.float32) - 0.25)
    random = random_signs(i * width + j, 0)
    return rng.choice([-1, 1]) * periodic + rng.choice([-1, 1]) * 0.75 * random
