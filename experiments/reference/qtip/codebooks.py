"""Independent NumPy definitions of the pinned QTIP computed codebooks."""

import numpy as np


def one_mad(states):
    x = np.asarray(states, dtype=np.uint64) & np.uint64(0xffffffff)
    x = (x * np.uint64(34038481) + np.uint64(76625530)) & np.uint64(0xffffffff)
    byte_sum = sum((x >> np.uint64(shift)) & np.uint64(255) for shift in (0, 8, 16, 24))
    return ((byte_sum.astype(np.float32) - np.float32(510)) / np.float32(147.800537109375))


def three_inst(states):
    x = np.asarray(states, dtype=np.uint64) & np.uint64(0xffffffff)
    x = x * np.uint64(89226354) + np.uint64(64248484)
    mask16 = (1 << 15) | ((1 << 12) - 1)
    mask32 = (mask16 << 16) | mask16
    bits = (x & np.uint64(mask32)) ^ np.uint64(996162400)
    lo = (bits & np.uint64(0xffff)).astype(np.uint16).view(np.float16)
    hi = ((bits >> np.uint64(16)) & np.uint64(0xffff)).astype(np.uint16).view(np.float16)
    return (lo + hi).astype(np.float32)
