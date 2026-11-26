"""Enumerate live generator DAGs and verify output equivalence."""

from collections import Counter
from functools import lru_cache
import hashlib
from itertools import product

import numpy as np

from experiments.generator_algebra.bitvector import Instruction, MASK, codebook, evaluate, scalar_instruction


COMMUTATIVE = {"ADD", "MUL", "XOR", "AND", "OR"}
CONSTANTS = (0, 1, 2, 3, 4, 5, 7, 8, 13, 16, 17, 24, 31, 33, 127, 255,
             0x01010101, 0x045D9F3B, 0x9E3779B9, 0xFFFFFFFF)


# Sign extraction, conversion to float, and scale multiplication per weight.
OUTPUT_COST = 3
SCALE_SETUP = 1  # One exponent-to-power-of-two conversion per block.


def cost(program, width):
    """Scalar DAG cost, excluding bitfield unpack and loop or index overhead."""
    depths = {"s": 0, "i": 0}
    live = {f"r{len(program) - 1}"}
    peak = 1
    for step in reversed(range(len(program))):
        live.discard(f"r{step}")
        live.update(arg for arg in program[step].args if isinstance(arg, str))
        peak = max(peak, len(live))
    for step, instruction in enumerate(program):
        depths[f"r{step}"] = 1 + max(depths.get(arg, 0) for arg in instruction.args)
    return {
        "instructions": len(program),
        "dependency_depth": depths[f"r{len(program) - 1}"],
        "peak_live_values": peak,
        "cost_per_block": width * (len(program) + OUTPUT_COST) + SCALE_SETUP,
    }


def is_identity(op, args):
    if len(args) == 1:
        return False
    a, b = args
    if a == b and op in {"SUB", "XOR", "AND", "OR"}:
        return True
    if op in {"ADD", "XOR", "OR"} and 0 in args:
        return True
    if op == "SUB" and b == 0 or op == "MUL" and (0 in args or 1 in args):
        return True
    if op == "AND" and (0 in args or MASK in args) or op == "OR" and MASK in args:
        return True
    return op in {"SHL", "SHR", "ROTL"} and (a == 0 or isinstance(b, int) and b % 32 == 0)


def choices(registers, operations, constants):
    names = ("s", "i", *(f"r{step}" for step in range(registers)))
    operands = names + tuple(constants)
    seen_constants = set(constants)
    for op in operations:
        unary = op in {"NOT", "POPCOUNT"}
        for args in product(operands, repeat=1 if unary else 2):
            if op in COMMUTATIVE and operands.index(args[0]) > operands.index(args[1]):
                continue
            if not any(isinstance(arg, str) for arg in args):
                # Retain a shortest derivation of each new literal. Computed
                # literals can feed later instructions, including dynamic shifts.
                value = scalar_instruction(op, args)
                if value in seen_constants:
                    continue
                seen_constants.add(value)
            elif is_identity(op, args) and (args[0] != args[-1]
                                            or op not in {"SUB", "XOR"} or 0 in constants):
                continue
            yield Instruction(op, args)


def live_instructions(program):
    live = {f"r{len(program) - 1}"}
    for step in reversed(range(len(program))):
        if f"r{step}" not in live:
            return False
        live.remove(f"r{step}")
        live.update(arg for arg in program[step].args if isinstance(arg, str))
    return True


def prefixes(length, options):
    return product(*options[:length - 1])


def extensions(prefix, options):
    """Enumerate connected DAGs, including forks and shared intermediate values."""
    length = len(prefix) + 1
    for instruction in options[length - 1]:
        if length > 1 and f"r{length - 2}" not in instruction.args:
            continue
        program = (*prefix, instruction)
        if live_instructions(program):
            yield program


class SemanticScreen:
    """Fingerprint signed outputs, verifying collisions across the configured domain."""

    def __init__(self, seed_bits, width):
        self.width = width
        self.seed_bits = seed_bits
        rng = np.random.default_rng(75414)
        seeds = rng.choice(1 << seed_bits, min(24, 1 << seed_bits), replace=False)
        self.probes = np.unique(np.concatenate(([0, (1 << seed_bits) - 1], seeds)))[:, None]
        self.indices = np.arange(width, dtype=np.uint32)
        self.representatives = {}
        self.materialize = lru_cache(maxsize=128)(self.full)
        self.audit = Counter()

    def full(self, program):
        return codebook(program, self.seed_bits, self.width)

    def accept(self, program):
        probe = np.broadcast_to(evaluate(program, self.probes, self.indices),
                                (len(self.probes), self.width))
        needs_seed = np.all(probe == probe[:1])
        needs_index = np.all(probe == probe[:, :1])
        if needs_seed or needs_index:
            full = self.materialize(program)
            if ((needs_seed and np.all(full == full[:1]))
                    or (needs_index and np.all(full == full[:, :1]))):
                self.audit["degenerate"] += 1
                return False
        key = hashlib.blake2b(probe.tobytes(), digest_size=16).digest()
        matches = self.representatives.setdefault(key, [])
        for representative in matches:
            self.audit["collision_checks"] += 1
            if np.array_equal(self.materialize(program), self.materialize(representative)):
                self.audit["equivalent"] += 1
                return False
        if matches:
            self.audit["distinct_collisions"] += 1
        matches.append(program)
        self.audit["distinct"] += 1
        return True
