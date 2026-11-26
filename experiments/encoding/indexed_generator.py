"""Nearest-neighbour encoding over a generated signed-byte codebook."""

import numpy as np
from scipy.spatial import cKDTree

from experiments.encoding import synthesized
from experiments.generator_algebra import bitvector


class DirectEncoder:
    """An exact Euclidean index of the frozen generator's full decoded space."""

    def __init__(self, program, seed_bits, threads):
        self.program = program
        self.seed_bits = seed_bits
        self.threads = threads
        self.unscaled = bitvector.codebook(program, seed_bits, 8)
        self.scales = np.arange(-8, 8)
        words = np.ldexp(self.unscaled[:, None, :].astype(np.float64), self.scales[None, :, None])
        unique, self.encodings = np.unique(words.reshape(-1, 8), axis=0, return_index=True)
        self.index = cKDTree(unique, leafsize=24)

    def encode(self, blocks):
        distance, positions = self.index.query(blocks, k=2, eps=0, workers=self.threads)
        winners = self.encodings[positions[:, 0]]
        # Match the exhaustive scorer's lowest-seed, then lowest-exponent tie break.
        tied = distance[:, 0] == distance[:, 1]
        if tied.any():
            radii = np.nextafter(distance[tied, 0], np.inf)
            candidates = self.index.query_ball_point(blocks[tied], radii, eps=0, workers=self.threads)
            for row, neighbors in zip(np.flatnonzero(tied), candidates):
                options = np.asarray(neighbors)
                errors = np.square(self.index.data[options] - blocks[row]).sum(axis=1)
                winners[row] = self.encodings[options[errors == errors.min()]].min()
        seeds, exponents = winners // 16, self.scales[winners % 16]
        return synthesized.pack(seeds, exponents, self.seed_bits)

    def decode(self, payload, count):
        bits = self.seed_bits + 4
        flat = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")[:count * bits]
        word = flat.reshape(count, bits) @ (1 << np.arange(bits))
        seeds = word & ((1 << self.seed_bits) - 1)
        exponents = ((word >> self.seed_bits) ^ 8) - 8
        return np.ldexp(self.unscaled[seeds].astype(np.float32), exponents[:, None])

    def encode_matrix(self, matrix):
        if matrix.shape[1] % 8:
            raise ValueError("Expected whole eight-weight blocks within each row")
        blocks = matrix.reshape(-1, 8)
        payloads = []
        restored = np.empty_like(blocks)
        for start in range(0, len(blocks), 32768):
            batch = blocks[start:start + 32768]
            payload = self.encode(batch)
            reconstruction = self.decode(payload, len(batch))
            if start == 0:
                count = min(16, len(batch))
                prefix = payload[:(count * (self.seed_bits + 4) + 7) // 8]
                np.testing.assert_array_equal(synthesized.decode(self.program, prefix, self.seed_bits, 8, count=count),
                                              reconstruction[:count])
            restored[start:start + len(batch)] = reconstruction
            payloads.append(payload)
        return restored.reshape(matrix.shape), b"".join(payloads)
