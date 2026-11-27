"""Small semantics-preserving rewrites for straight-line bitvector programs."""

from collections import Counter

from experiments.generator_algebra.bitvector import Instruction


def _mul_parts(instruction):
    if instruction.op != "MUL":
        return None
    left, right = instruction.args
    if isinstance(left, int) and isinstance(right, str):
        return right, left
    if isinstance(right, int) and isinstance(left, str):
        return left, right
    return None


def fold_multipliers(program):
    """Fold adjacent multiply constants whose intermediate has one consumer."""
    uses = Counter(arg for instruction in program for arg in instruction.args
                   if isinstance(arg, str))
    result = []
    names = {"s": "s", "i": "i"}
    folds = 0
    for step, instruction in enumerate(program):
        args = tuple(names.get(arg, arg) if isinstance(arg, str) else arg
                     for arg in instruction.args)
        current = Instruction(instruction.op, args)
        parts = _mul_parts(current)
        if parts and result and step > 0 and uses[f"r{step - 1}"] == 1:
            previous = _mul_parts(result[-1])
            if previous and parts[0] == f"r{len(result) - 1}":
                result.pop()
                current = Instruction("MUL", (previous[0],
                                              (previous[1] * parts[1]) & 0xffffffff))
                folds += 1
        names[f"r{step}"] = f"r{len(result)}"
        result.append(current)
    return tuple(result), folds
