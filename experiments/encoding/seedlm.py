"""Seeded subspace search and a packed reference decoder."""

from dataclasses import dataclass

import numpy as np

from experiments.encoding.pursuit import sse
from experiments.generator_algebra.dictionary import hash32

TAPS = {3: (0, 1), 4: (0, 1), 5: (0, 2), 6: (0, 1), 8: (0, 2, 3, 4),
        12: (0, 1, 2, 8), 16: (0, 1, 3, 12)}
FAMILIES = ("lfsr", "independent_lfsr", "xorshift", "splitmix", "counter", "hadamard", "mixed")
SIGN_FAMILIES = ("binary", "ternary")


@dataclass(frozen=True)
class Config:
    block: int
    latent: int
    seed_bits: int = 16
    coefficient_bits: int = 4
    family: str = "lfsr"

    def __post_init__(self):
        if not (1 <= self.block <= 64 and 1 <= self.latent <= 8
                and 2 <= self.coefficient_bits <= 8 and 2 <= self.seed_bits <= 16
                and self.family in FAMILIES + SIGN_FAMILIES):
            raise ValueError("Invalid seeded subspace configuration")
        if self.family in ("lfsr", "mixed") + SIGN_FAMILIES and self.seed_bits not in TAPS:
            raise ValueError(f"No LFSR taps configured for width {self.seed_bits}")
        if self.family == "independent_lfsr" and any(
            width not in TAPS for width, _ in seed_fields(self)
        ):
            raise ValueError("Each independent seed field needs a supported LFSR width")

    @property
    def payload_bits(self):
        return self.seed_bits + 4 + self.latent * self.coefficient_bits

    @property
    def rate(self):
        return self.payload_bits / self.block


def lfsr(seeds, count, width):
    mask = sum(1 << position for position in TAPS[width])
    state = np.asarray(seeds, dtype=np.uint32)
    values = np.empty((len(state), count), dtype=np.float32)
    for index in range(count):
        feedback = (np.bitwise_count(state & mask).astype(np.uint32) & 1) << (width - 1)
        state = (state >> 1) | feedback
        values[:, index] = state
    midpoint = 1 << (width - 1)
    return (values - midpoint) / (midpoint - 1)


def seed_ids(config):
    ids = np.arange(1, 1 << config.seed_bits, dtype=np.uint32)
    if config.family == "independent_lfsr":
        valid = np.ones(len(ids), dtype=bool)
        for width, shift in seed_fields(config):
            valid &= ((ids >> shift) & ((1 << width) - 1)) != 0
        ids = ids[valid]
    return ids


def seed_fields(config):
    width, extra = divmod(config.seed_bits, config.latent)
    shift = 0
    for column in range(config.latent):
        bits = width + (column < extra)
        yield bits, shift
        shift += bits

PUBLISHED = (Config(12, 4), Config(8, 3))


def splitmix64(word):
    word = (word ^ (word >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    word = (word ^ (word >> 27)) * np.uint64(0x94D049BB133111EB)
    return word ^ (word >> 31)


def signed_unit(words):
    return (words.astype(np.float32) / np.float32(1 << 31)) - 1


def basis(config, seeds):
    """Regenerate each C×P basis from the stored seed or seed fields."""
    seeds = np.asarray(seeds, dtype=np.uint32)
    rows, columns = config.block, config.latent
    family = config.family
    if family == "independent_lfsr":
        return np.stack([
            lfsr((seeds >> shift) & ((1 << width) - 1), rows, width)
            for width, shift in seed_fields(config)
        ], axis=2)
    if family in SIGN_FAMILIES:
        state = seeds
        mask = sum(1 << position for position in TAPS[config.seed_bits])
        nonzero_bit = 1 << (config.seed_bits // 2 - 1)
        values = np.empty((len(seeds), rows * columns), dtype=np.float32)
        for index in range(rows * columns):
            feedback = np.bitwise_count(state & mask).astype(np.uint32) & 1
            state = (state >> 1) | (feedback << (config.seed_bits - 1))
            sign = 1 - 2 * (state & 1).astype(np.float32)
            values[:, index] = sign if family == "binary" else sign * ((state & nonzero_bit) != 0)
    elif family == "lfsr":
        values = lfsr(seeds, rows * columns, config.seed_bits)
    elif family == "xorshift":
        state = hash32(seeds)
        values = np.empty((len(seeds), rows * columns), dtype=np.float32)
        for index in range(rows * columns):
            state = state ^ (state << 13)
            state = state ^ (state >> 17)
            state = state ^ (state << 5)
            values[:, index] = signed_unit(state)
    elif family == "splitmix":
        counters = seeds[:, None].astype(np.uint64) << 32
        counters = (counters | np.arange(rows * columns, dtype=np.uint64)) + np.uint64(0x9E3779B97F4A7C15)
        values = signed_unit(splitmix64(counters) >> 32)
    elif family == "counter":
        counters = (seeds[:, None] << 16) | np.arange(rows * columns, dtype=np.uint32)
        values = signed_unit(hash32(counters))
    elif family in ("hadamard", "mixed"):
        index = np.arange(rows, dtype=np.uint32)[None, :]
        mask = 1 - 2 * (hash32((seeds[:, None] << 16) | index) & 1).astype(np.float32)
        shifts = hash32(seeds ^ np.uint32(0xC001C0DE)) & ((1 << (rows - 1).bit_length()) - 1)
        walsh = (np.arange(columns, dtype=np.uint32)[None, :] ^ shifts[:, None])[:, None, :]
        signs = 1 - 2 * (np.bitwise_count(index[:, :, None] & walsh) & 1).astype(np.float32)
        result = signs * mask[:, :, None]
        if family == "mixed":
            result[:, :, 1::2] = lfsr(seeds, rows * columns, config.seed_bits).reshape(
                -1, rows, columns
            )[:, :, 1::2]
        return result
    else:
        raise ValueError(f"Unknown generator family {family}")
    return values.reshape(-1, rows, columns)


def coefficient_codes(coefficients, bits, rule):
    maximum = np.abs(coefficients).max(axis=1, keepdims=True)
    exponent = np.frexp(maximum)[1] - 1
    if rule == "range":
        # Use enough mantissa range to fit the largest coefficient.
        positive = np.maximum(coefficients.max(axis=1, keepdims=True), 0)
        negative = np.maximum(-coefficients.min(axis=1, keepdims=True), 0)
        required = np.maximum(positive / ((1 << (bits - 1)) - 1),
                              negative / (1 << (bits - 1)))
        mantissa, exponent = np.frexp(required)
        exponent -= mantissa == 0.5
    exponent = np.where(maximum > 0, np.clip(exponent, -8, 7), -8)
    scale = np.exp2(exponent.astype(np.float32))
    integers = np.clip(np.rint(coefficients / scale), -(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    return integers, exponent, scale


def greedy_fit(bases, blocks):
    # Successive projections in a fixed basis. Each column gets one update.
    residual = np.broadcast_to(blocks.T, (len(bases), *blocks.T.shape)).copy()
    coefficients = np.zeros((len(bases), bases.shape[-1], len(blocks)), dtype=np.float32)
    available = np.ones_like(coefficients, dtype=bool)
    norms = np.sum(bases ** 2, axis=1)[:, :, None]
    trials, samples = np.ogrid[:len(bases), :len(blocks)]
    for _ in range(bases.shape[-1]):
        dots = bases.transpose(0, 2, 1) @ residual
        projection = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)
        chosen = np.argmax(np.where(available, projection * dots, -1), axis=1)
        value = projection[trials, chosen, samples]
        coefficients[trials, chosen, samples] = value
        available[trials, chosen, samples] = False
        residual -= bases.transpose(0, 2, 1)[trials, chosen, :].transpose(0, 2, 1) * value[:, None, :]
    return coefficients

def refine_codes(bases, blocks, coefficients, bits):
    """Try three exponents and one integer coordinate-descent sweep per exponent."""
    _, starting_exponent, _ = coefficient_codes(coefficients, bits, "range")
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    gram = bases.transpose(0, 2, 1) @ bases
    dots = bases.transpose(0, 2, 1) @ blocks.T
    best_error = np.full((len(bases), len(blocks)), np.inf, dtype=np.float32)
    best_integers = np.empty_like(coefficients)
    best_exponent = np.empty_like(starting_exponent)
    for offset in (0, -1, 1):
        exponents = np.clip(starting_exponent + offset, -8, 7)
        scale = np.exp2(exponents.astype(np.float32))
        code = np.clip(np.rint(coefficients / scale), lo, hi)
        normalized_dots = dots / scale
        for column in range(bases.shape[-1]):
            diagonal = gram[:, column, column, None]
            residual_dot = normalized_dots[:, column] - (gram[:, column:column + 1] @ code)[:, 0]
            change = np.divide(residual_dot, diagonal, out=np.zeros_like(residual_dot), where=diagonal > 0)
            code[:, column] = np.clip(np.rint(code[:, column] + change), lo, hi)
        residual = bases @ (code * scale) - blocks.T
        errors = np.einsum("sci,sci->si", residual, residual)
        improved = errors < best_error
        np.copyto(best_error, errors, where=improved)
        np.copyto(best_integers, code, where=improved[:, None, :])
        np.copyto(best_exponent, exponents, where=improved[:, None, :])
    return best_integers, best_exponent, np.exp2(best_exponent.astype(np.float32))


def encode_candidates(config, blocks, seeds, rules, fitting="joint"):
    """Yield winners in one candidate batch, with coefficients shaped (blocks, P)."""
    bases = basis(config, seeds)
    if fitting == "greedy":
        coefficients = greedy_fit(bases, blocks)
    else:
        # SVD in float64 handles exact rank deficiencies in discrete codebooks.
        inverse = np.linalg.pinv(bases.astype(np.float64), rcond=1e-12).astype(np.float32)
        coefficients = inverse @ blocks.T
    columns = np.arange(len(blocks))
    for rule in rules:
        if rule == "float":
            integers, exponent, decoded = coefficients, None, coefficients
        else:
            if rule == "refine":
                integers, exponent, scale = refine_codes(bases, blocks, coefficients, config.coefficient_bits)
            else:
                integers, exponent, scale = coefficient_codes(coefficients, config.coefficient_bits, rule)
            decoded = integers * scale
        residual = bases @ decoded
        residual -= blocks.T
        errors = np.einsum("sci,sci->si", residual, residual)
        winners = errors.argmin(axis=0)
        exponents = None if exponent is None else exponent[winners, 0, columns].astype(np.int8)
        yield rule, (errors[winners, columns], seeds[winners],
                     integers.transpose(0, 2, 1)[winners, columns].copy(), exponents)


def search(config, blocks, rules=("paper",), fitting="joint", prefixes=(), limit=None, seed_batch=2048):
    """Exhaustively search a nested sequence of seed prefixes and retain packed fields."""
    if fitting not in ("joint", "greedy") or not rules or set(rules) - {"paper", "range", "float", "refine"}:
        raise ValueError("Choose supported fitting and coefficient rules")
    if limit is not None and limit < 1:
        raise ValueError("Seed limit must be positive")
    blocks = np.asarray(blocks, dtype=np.float32)
    seeds = np.random.default_rng(49789).permutation(seed_ids(config))
    if limit is not None:
        seeds = seeds[:limit]
    boundaries = sorted(set(prefixes) | {len(seeds)})
    if (blocks.ndim != 2 or blocks.shape[1] != config.block or not len(blocks)
            or not np.isfinite(blocks).all() or seed_batch < 1):
        raise ValueError("Expected one or more blocks matching the codebook")
    if not all(0 < count <= len(seeds) for count in boundaries):
        raise ValueError("Each prefix must contain a valid number of seeds")
    best = {}
    previous = 0
    for end in boundaries:
        for start in range(previous, end, seed_batch):
            candidates = seeds[start:min(start + seed_batch, end)]
            for name, current in encode_candidates(config, blocks, candidates, rules, fitting):
                if name not in best:
                    best[name] = current
                    continue
                saved = best[name]
                better = current[0] < saved[0]
                for destination, source in zip(saved, current):
                    if destination is not None:
                        destination[better] = source[better]
        for name, result in best.items():
            yield end, name, tuple(None if field is None else field.copy() for field in result)
        previous = end


def pack(config, seeds, coefficients, exponents):
    """Pack consecutive little-endian bit fields without per-block padding."""
    seeds = np.asarray(seeds)
    coefficients = np.asarray(coefficients)
    exponents = np.asarray(exponents)
    limit = 1 << (config.coefficient_bits - 1)
    if (seeds.shape != exponents.shape or coefficients.shape != (len(seeds), config.latent)
            or np.any((seeds < 1) | (seeds >= 1 << config.seed_bits))
            or np.any((coefficients < -limit) | (coefficients >= limit))
            or np.any((exponents < -8) | (exponents > 7))
            or not all(np.all(values == np.trunc(values)) for values in (seeds, coefficients, exponents))):
        raise ValueError("Invalid codebook fields")
    validate_seed_fields(config, seeds)
    fields = [(seeds, config.seed_bits), (exponents, 4)]
    fields.extend((values, config.coefficient_bits) for values in coefficients.T)
    bits = np.concatenate([
        (values.astype(np.int64)[:, None] >> np.arange(width)) & 1 for values, width in fields
    ], axis=1)
    return np.packbits(bits.ravel(), bitorder="little").tobytes()


def unpack(config, payload, count):
    """Read exactly count blocks. Terminal padding is at most seven bits."""
    if count < 0 or len(payload) != (count * config.payload_bits + 7) // 8:
        raise ValueError("Payload length does not match block count")
    flat = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    if np.any(flat[count * config.payload_bits:]):
        raise ValueError("Nonzero terminal padding")
    bits = flat[:count * config.payload_bits].reshape(count, config.payload_bits)

    def read(start, width, signed=False):
        value = bits[:, start:start + width] @ (1 << np.arange(width))
        if signed:
            sign = 1 << (width - 1)
            value = (value ^ sign) - sign
        return value

    seeds = read(0, config.seed_bits).astype(np.uint32)
    exponents = read(config.seed_bits, 4, signed=True)
    first = config.seed_bits + 4
    coefficients = np.column_stack([
        read(first + index * config.coefficient_bits, config.coefficient_bits, signed=True)
        for index in range(config.latent)
    ])
    if np.any(seeds == 0):
        raise ValueError("Zero is a reserved seed")
    validate_seed_fields(config, seeds)
    return seeds, coefficients, exponents


def validate_seed_fields(config, seeds):
    if config.family == "independent_lfsr" and any(
        np.any(((seeds.astype(np.uint32) >> shift) & ((1 << width) - 1)) == 0)
        for width, shift in seed_fields(config)
    ):
        raise ValueError("Zero is reserved in each independent seed field")


def decode(config, payload, count):
    seeds, coefficients, exponents = unpack(config, payload, count)
    scales = np.exp2(exponents.astype(np.float32))
    return np.einsum("ncp,np->nc", basis(config, seeds), coefficients.astype(np.float32)) * scales[:, None]


def check_reconstruction(config, blocks, result):
    errors, seeds, coefficients, exponents = result
    payload = pack(config, seeds, coefficients, exponents)
    reconstructions = decode(config, payload, len(blocks))
    np.testing.assert_allclose(sse(blocks, reconstructions), errors, rtol=4e-5, atol=2e-8)
    return payload
