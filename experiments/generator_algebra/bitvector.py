"""Scalar and vector semantics for straight-line uint32 generator programs."""

from dataclasses import dataclass

import numpy as np


OPERATIONS = {
    "ADD": np.add, "SUB": np.subtract, "MUL": np.multiply,
    "XOR": np.bitwise_xor, "AND": np.bitwise_and, "OR": np.bitwise_or,
    "NOT": np.bitwise_not, "SHL": np.left_shift, "SHR": np.right_shift,
    "ROTL": None, "POPCOUNT": None,
}
MASK = (1 << 32) - 1


@dataclass(frozen=True)
class Instruction:
    op: str
    args: tuple

    def __str__(self):
        return f"{self.op}({', '.join(str(arg) if isinstance(arg, str) else hex(arg) for arg in self.args)})"


def vector_instruction(instruction, values):
    op = instruction.op
    args = [values[arg] if isinstance(arg, str) else np.uint32(arg) for arg in instruction.args]
    if op == "POPCOUNT":
        return np.bitwise_count(args[0]).astype(np.uint32)
    if op in ("SHL", "SHR", "ROTL"):
        word, shift = args[0], args[1] & np.uint32(31)
        if op == "ROTL":
            return (word << shift) | (word >> (np.subtract(np.uint32(0), shift) & np.uint32(31)))
        return OPERATIONS[op](word, shift)
    return OPERATIONS[op](*args)


def evaluate(program, seed, index):
    values = {"s": np.asarray(seed, dtype=np.uint32), "i": np.asarray(index, dtype=np.uint32)}
    for step, instruction in enumerate(program):
        values[f"r{step}"] = vector_instruction(instruction, values)
    return values[f"r{len(program) - 1}"].astype(np.uint8).view(np.int8)


def codebook(program, seed_bits, width):
    seeds = np.arange(1 << seed_bits, dtype=np.uint32)
    indices = np.arange(width, dtype=np.uint32)
    return np.broadcast_to(evaluate(program, seeds[:, None], indices), (len(seeds), width))


def expression(program):
    return "; ".join(f"r{step}={instruction}" for step, instruction in enumerate(program))


def scalar_instruction(op, args):
    x = args[0]
    if op == "NOT":
        return (~x) & MASK
    if op == "POPCOUNT":
        return x.bit_count()
    y = args[1]
    shifts = y % 32
    if op == "ROTL":
        return ((x << shifts) | (x >> ((32 - shifts) % 32))) & MASK
    match op:
        case "ADD": return (x + y) & MASK
        case "SUB": return (x - y) & MASK
        case "MUL": return (x * y) & MASK
        case "XOR": return x ^ y
        case "AND": return x & y
        case "OR": return x | y
        case "SHL": return (x << shifts) & MASK
        case "SHR": return x >> shifts
    raise ValueError(f"Unknown operation {op}")
