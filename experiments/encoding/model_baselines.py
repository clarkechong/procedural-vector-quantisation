"""Scalar and shortlisted SeedLM codecs for model evaluation."""

from itertools import combinations

import numpy as np

from experiments.encoding.seedlm import PUBLISHED, basis, decode, pack, seed_ids


class Shortlist:
    """Use continuous residual to select a fixed number of seeds for discrete fitting."""
    def __init__(self, count, keep):
        self.config = PUBLISHED[1]
        self.seeds = np.random.default_rng(49789).permutation(seed_ids(self.config))[:count]
        self.bases = basis(self.config, self.seeds)
        u, s, _ = np.linalg.svd(self.bases.astype(np.float64), full_matrices=False)
        u *= (s > s[:, :1] * 1e-13)[:, None, :]
        self.orthogonal = np.ascontiguousarray(u.transpose(0, 2, 1).astype(np.float32))
        self.projection = self.orthogonal.reshape(-1, 8)
        self.inverse = np.linalg.pinv(self.bases.astype(np.float64)).astype(np.float32)
        self.gram = self.bases.transpose(0, 2, 1) @ self.bases
        self.keep = min(keep, count)

    def candidates(self, blocks):
        projection = (blocks @ self.projection.T).reshape(len(blocks), len(self.seeds), 3)
        score = np.einsum("bsp,bsp->bs", projection, projection)
        return np.argpartition(score, -self.keep, axis=1)[:, -self.keep:]

    def quantize(self, blocks, rule):
        count = len(blocks)
        selected = self.candidates(blocks)
        bases = self.bases[selected]
        coefficients = (self.inverse[selected] @ blocks[:, None, :, None])[..., 0]
        if rule == "paper":
            exponent = np.frexp(np.abs(coefficients).max(axis=-1))[1] - 1
        else:
            required = np.maximum(coefficients.max(axis=-1) / 7, -coefficients.min(axis=-1) / 8)
            fraction, exponent = np.frexp(required)
            exponent -= fraction == 0.5
        exponent = np.clip(exponent, -8, 7)
        if rule == "refine":
            dots = (bases.swapaxes(-1, -2) @ blocks[:, None, :, None])[..., 0]
            gram = self.gram[selected]
        rows = np.arange(count)
        best_error = np.full(count, np.inf)
        best_seed = np.zeros(count, np.uint32)
        best_code = np.zeros((count, 3), np.int8)
        best_e = np.zeros(count, np.int8)
        for offset in ((0, -1, 1) if rule == "refine" else (0,)):
            exponents = np.clip(exponent + offset, -8, 7)
            scale = np.exp2(exponents.astype(np.float32))
            code = np.clip(np.rint(coefficients / scale[..., None]), -8, 7)
            if rule == "refine":
                for col in range(3):
                    residual = dots[..., col] / scale - (gram[..., col, :] * code).sum(axis=-1)
                    diagonal = gram[..., col, col]
                    change = np.divide(residual, diagonal, out=np.zeros_like(residual), where=diagonal > 0)
                    code[..., col] = np.clip(np.rint(code[..., col] + change), -8, 7)
            reconstructed = (bases @ code[..., None])[..., 0] * scale[..., None]
            residual = reconstructed - blocks[:, None, :]
            errors = np.einsum("bsi,bsi->bs", residual, residual)
            winner = errors.argmin(axis=1)
            update = errors[rows, winner] < best_error
            best_error[update] = errors[rows, winner][update]
            best_seed[update] = self.seeds[selected[rows, winner]][update]
            best_code[update] = code[rows, winner][update]
            best_e[update] = exponents[rows, winner][update]
        words = best_seed.copy()
        words |= (best_e.astype(np.uint32) & 15) << 16
        for col in range(3):
            words |= (best_code[:, col].astype(np.uint32) & 15) << (20 + 4 * col)
        return words.astype("<u4").tobytes(), (best_seed, best_code, best_e)


class IndexedShortlist(Shortlist):
    """Index the full seed namespace by sign and top-two input coordinates."""
    def __init__(self, neighborhood, keep):
        super().__init__(65535, keep)
        self.pair_id = np.zeros((8, 8), dtype=np.int64)
        centers = np.full((28, 8), 0.23, dtype=np.float32)
        for index, (left, right) in enumerate(combinations(range(8), 2)):
            self.pair_id[left, right] = self.pair_id[right, left] = index
            centers[index, [left, right]] = 0.61
        self.sign_bits = 1 << np.arange(8, dtype=np.int64)
        signs = 1 - 2 * ((np.arange(256)[:, None] & self.sign_bits) != 0)
        centers = (signs[:, None, :] * centers).astype(np.float32).reshape(-1, 8)
        self.index = np.empty((len(centers), neighborhood), dtype=np.uint16)
        for start in range(0, len(centers), 32):
            batch = centers[start:start + 32]
            projected = (batch @ self.projection.T).reshape(len(batch), len(self.seeds), 3)
            energy = np.einsum("bsp,bsp->bs", projected, projected)
            self.index[start:start + len(batch)] = np.argpartition(
                energy, -neighborhood, axis=1)[:, -neighborhood:]

    def candidates(self, blocks):
        top_two = np.argpartition(np.abs(blocks), -2, axis=1)[:, -2:]
        keys = (np.signbit(blocks) @ self.sign_bits) * 28 + self.pair_id[tuple(top_two.T)]
        neighbors = self.index[keys]
        projected = (self.orthogonal[neighbors] @ blocks[:, None, :, None])[..., 0]
        scores = np.einsum("bsp,bsp->bs", projected, projected)
        local = np.argpartition(scores, -self.keep, axis=1)[:, -self.keep:]
        return np.take_along_axis(neighbors, local, axis=1)


def quantize_matrix(matrix, method, shortlist):
    if method == "int4_group64":
        blocks = matrix.reshape(-1, 64)
        scale = (np.max(np.abs(blocks), axis=1, keepdims=True) / 7).astype(np.float16)
        scale_f32 = scale.astype(np.float32)
        values = np.divide(blocks, scale_f32, out=np.zeros_like(blocks), where=scale_f32 > 0)
        integers = np.clip(np.rint(values), -8, 7).astype(np.int8)
        codes = integers.astype(np.uint8) & np.uint8(15)
        packed = codes[:, ::2] | (codes[:, 1::2] << 4)
        fields = np.concatenate((scale.view(np.uint8).reshape(-1, 2), packed), axis=1)
        payload = fields.tobytes()
        codes = np.frombuffer(payload, np.uint8).reshape(-1, 34)
        scales = codes[:, :2].copy().view("<f2").astype(np.float32)
        lower, upper = codes[:, 2:] & 15, codes[:, 2:] >> 4
        unpacked = np.stack((lower, upper), axis=-1).reshape(-1, 64)
        return (((unpacked.astype(np.int8) ^ 8) - 8) * scales).reshape(matrix.shape), payload

    blocks = matrix.reshape(-1, 8)
    payloads = []
    decoded = np.empty_like(blocks)
    for start in range(0, len(blocks), 4096):
        batch = blocks[start:start + 4096]
        payload, fields = shortlist.quantize(batch, "refine" if method == "indexed_refine" else method)
        if start == 0 and payload != pack(shortlist.config, *fields):
            raise RuntimeError("Payload must match reference serialization")
        payloads.append(payload)
        decoded[start:start + len(batch)] = decode(shortlist.config, payload, len(batch))
    return decoded.reshape(matrix.shape), b"".join(payloads)
