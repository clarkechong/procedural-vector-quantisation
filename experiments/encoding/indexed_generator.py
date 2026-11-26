"""Nearest-neighbour encoding over a generated signed-byte codebook."""

import numpy as np
from scipy.spatial import cKDTree

from experiments.encoding import synthesized
from experiments.generator_algebra import bitvector


class DirectEncoder:
    """An exact Euclidean index of the frozen generator's full decoded space."""

    def __init__(self, program, seed_bits, threads):
        if not 1 <= seed_bits <= 20:
            raise ValueError("Seed field must be 1–20 bits")
        self.program = program
        self.seed_bits = seed_bits
        self.threads = threads
        self.unscaled = bitvector.codebook(program, seed_bits, 8)
        self.scales = np.arange(-8, 8)
        self.separate_scales = seed_bits > 16
        if self.separate_scales:
            # A 20-bit seed has 16 million scaled encodings. Index unscaled
            # codewords once and query each power of two to keep RAM bounded.
            unique, self.encodings = np.unique(self.unscaled, axis=0, return_index=True)
            self.index = cKDTree(unique, leafsize=24)
            return
        words = np.ldexp(self.unscaled[:, None, :].astype(np.float64), self.scales[None, :, None])
        unique, self.encodings = np.unique(words.reshape(-1, 8), axis=0, return_index=True)
        self.index = cKDTree(unique, leafsize=24)

    def encode(self, blocks):
        if self.separate_scales:
            return self.encode_separate_scales(blocks)
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

    def encode_separate_scales(self, blocks):
        blocks = np.asarray(blocks, dtype=np.float64)
        best_error = np.full(len(blocks), np.inf)
        best_word = np.full(len(blocks), np.iinfo(np.int64).max, dtype=np.int64)
        for slot, exponent in enumerate(self.scales):
            targets = np.ldexp(blocks, -int(exponent))
            distance, positions = self.index.query(targets, k=2, eps=0, workers=self.threads)
            selected = positions[:, 0].copy()
            tied = distance[:, 0] == distance[:, 1]
            if tied.any():
                radii = np.nextafter(distance[tied, 0], np.inf)
                candidates = self.index.query_ball_point(targets[tied], radii, eps=0,
                                                        workers=self.threads)
                for row, neighbors in zip(np.flatnonzero(tied), candidates):
                    options = np.asarray(neighbors)
                    errors = np.square(self.index.data[options] - targets[row]).sum(axis=1)
                    equal = options[errors == errors.min()]
                    selected[row] = equal[np.argmin(self.encodings[equal])]
            reconstruction = np.ldexp(self.index.data[selected], int(exponent))
            errors = np.square(blocks - reconstruction).sum(axis=1)
            words = self.encodings[selected] * 16 + slot
            improve = (errors < best_error) | ((errors == best_error) & (words < best_word))
            best_error[improve] = errors[improve]
            best_word[improve] = words[improve]
        return synthesized.pack(best_word // 16, self.scales[best_word % 16], self.seed_bits)

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
