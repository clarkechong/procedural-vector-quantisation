"""Native SeedLM paper encoder with optional all-seed projection bounds."""

import ctypes
from itertools import combinations

import numpy as np

from experiments.encoding.seedlm import basis, decode, pack, seed_ids


class PaperEncoder:
    """The paper's coefficient rule with all seeds or a projection-based seed index."""

    def __init__(self, config, pool, library, threads, projection_bound=False):
        self.config = config
        self.threads = threads
        self.pool = pool
        self.load = ctypes.CDLL(str(library))
        self.native = self.load.paper_seedlm_encode
        ptr = ctypes.c_void_p
        self.native.argtypes = [ctypes.c_int, ptr, ctypes.c_size_t, ptr, ptr, ptr, ptr,
                                ctypes.c_size_t, ctypes.c_size_t, ptr, ptr, ptr]
        self.native.restype = None
        bases = basis(config, seed_ids(config))
        self.bases = np.ascontiguousarray(bases.transpose(0, 2, 1))
        self.inverse = np.linalg.pinv(bases.astype(np.float64), rcond=1e-12).astype(np.float32)
        self.index = None
        self.orthogonal = None
        projection_bound = projection_bound and pool < 65535
        if projection_bound:
            try:
                self.project = self.load.paper_seedlm_project
            except AttributeError as error:
                raise ValueError("Build the native encoder with CBLAS or use --pool 65535 for unpruned search") from error
            self.project.argtypes = [ctypes.c_int, ptr, ctypes.c_size_t, ptr, ptr, ptr, ctypes.c_size_t,
                                     ptr, ptr, ptr]
            self.project.restype = None
        if pool < 65535 or projection_bound:
            u, singular, _ = np.linalg.svd(bases.astype(np.float64), full_matrices=False)
            if projection_bound:
                # Retaining all columns makes the bound safe for rank-deficient bases.
                self.orthogonal = np.ascontiguousarray(u.transpose(0, 2, 1).astype(np.float32))
            if pool < 65535:
                u *= (singular > singular[:, :1] * 1e-12)[:, None, :]
                self.build_index(u)

    def build_index(self, u):
        width = self.config.block
        self.sign_bits = 1 << np.arange(width, dtype=np.int32)
        signs = 1 - 2 * ((np.arange(1 << width)[:, None] & self.sign_bits) != 0)
        if width == 8:
            self.shape_positions = list(combinations(range(width), 2))
            self.pair_id = np.zeros((width, width), dtype=np.int32)
            for k, (left, right) in enumerate(self.shape_positions):
                self.pair_id[left, right] = self.pair_id[right, left] = k
        else:
            self.shape_positions = [tuple(range(start, start + 4)) for start in range(0, width, 4)]
        templates = np.full((len(self.shape_positions), width), .35, dtype=np.float32)
        for row, positions in enumerate(self.shape_positions):
            templates[row, list(positions)] = .90 if width == 8 else .55
        centers = (signs[:, None, :] * templates).astype(np.float32).reshape(-1, width)
        projection = np.ascontiguousarray(u.transpose(0, 2, 1).astype(np.float32)).reshape(-1, width)
        self.index = np.empty((len(centers), self.pool), dtype=np.uint16)
        for start in range(0, len(centers), 32):
            batch = centers[start:start + 32]
            projected = (batch @ projection.T).reshape(len(batch), len(self.bases), self.config.latent)
            scores = np.einsum("bsp,bsp->bs", projected, projected)
            ids = np.argpartition(scores, -self.pool, axis=1)[:, -self.pool:]
            # Deterministic order with nearby basis addresses together.
            self.index[start:start + len(batch)] = np.sort(ids, axis=1)

    def candidate_keys(self, blocks):
        sign = np.signbit(blocks) @ self.sign_bits
        if self.config.block == 8:
            indices = np.argpartition(np.abs(blocks), -2, axis=1)[:, -2:]
            shape = self.pair_id[tuple(indices.T)]
        else:
            shape = np.square(blocks).reshape(-1, 3, 4).sum(axis=-1).argmax(axis=1)
        return (sign * len(self.shape_positions) + shape).astype(np.int32)

    def fields(self, blocks, *, exhaustive=False, use_bound=True):
        blocks = np.ascontiguousarray(blocks, dtype=np.float32)
        if blocks.ndim != 2 or blocks.shape[1] != self.config.block or not len(blocks):
            raise ValueError("Expected nonempty blocks of the published width")
        use_index = not exhaustive and self.index is not None
        keys = self.candidate_keys(blocks) if use_index else None
        seeds = np.empty(len(blocks), dtype=np.uint32)
        coefficients = np.empty((len(blocks), self.config.latent), dtype=np.int8)
        exponents = np.empty(len(blocks), dtype=np.int8)
        self.native(self.config.block, blocks.ctypes.data, len(blocks), self.inverse.ctypes.data,
                    self.bases.ctypes.data, self.index.ctypes.data if use_index else None,
                    keys.ctypes.data if use_index else None, self.pool if use_index else 65535,
                    self.threads, seeds.ctypes.data, coefficients.ctypes.data, exponents.ctypes.data)
        if use_bound and not exhaustive and self.orthogonal is not None:
            self.project(self.config.block, blocks.ctypes.data, len(blocks), self.inverse.ctypes.data,
                         self.bases.ctypes.data, self.orthogonal.ctypes.data, self.threads,
                         seeds.ctypes.data, coefficients.ctypes.data, exponents.ctypes.data)
        return seeds, coefficients, exponents

    def encode_matrix(self, matrix):
        width = self.config.block
        flat = matrix.ravel()
        # Llama 2 row widths are not divisible by 12. Block the contiguous
        # matrix stream and count any zero padding in the stored payload.
        padding = -flat.size % width
        if padding:
            flat = np.pad(flat, (0, padding))
        blocks = flat.reshape(-1, width)
        payloads = []
        restored = np.empty_like(blocks)
        for start in range(0, len(blocks), 16384):
            batch = blocks[start:start + 16384]
            payload = pack(self.config, *self.fields(batch))
            restored[start:start + len(batch)] = decode(self.config, payload, len(batch))
            payloads.append(payload)
        return restored.ravel()[:matrix.size].reshape(matrix.shape), b"".join(payloads)
