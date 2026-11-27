"""Independent implementation of pinned QTIP's two-pass tail-biting Viterbi rule."""

import csv
import json
from pathlib import Path
import time

import numpy as np

from experiments.reference.qtip.codebooks import one_mad, three_inst


def viterbi(row, mapping, state_bits=16, stride=2, overlap=None):
    target = np.asarray(row, dtype=np.float16).astype(np.float32)
    values = np.asarray(mapping, dtype=np.float32)
    states = 1 << state_bits
    low_bits = state_bits - stride
    groups = 1 << low_bits
    if len(values) != states:
        raise ValueError("codebook size mismatch")
    costs = (values - target[0]) ** 2
    if overlap is not None:
        allowed = np.arange(1 << stride) + (int(overlap) << stride)
        mask = np.full(states, np.inf, dtype=np.float32)
        mask[allowed] = 0
        costs += mask
    history = np.empty((len(target)-1, groups), dtype=np.uint8)
    columns = np.arange(groups)
    for position in range(1, len(target)):
        predecessor = costs.reshape(1 << stride, groups)
        best = predecessor.argmin(0)
        history[position-1] = best
        costs = np.repeat(predecessor[best, columns], 1 << stride) + (values - target[position]) ** 2
    if overlap is not None:
        allowed = int(overlap) + (np.arange(1 << stride) << low_bits)
        mask = np.full(states, np.inf, dtype=np.float32)
        mask[allowed] = 0
        costs += mask
    state = int(costs.argmin())
    path = np.empty(len(target), dtype=np.int32)
    path[-1] = state
    for position in range(len(target)-2, -1, -1):
        group = state >> stride
        state = group | (int(history[position, group]) << low_bits)
        path[position] = state
    return path


def quantize(row, mapping, state_bits=16, stride=2):
    row = np.asarray(row)
    half = len(row)//2
    first = viterbi(np.roll(row, half), mapping, state_bits, stride)
    overlap = int(first[half] >> stride)
    path = viterbi(row, mapping, state_bits, stride, overlap)
    low_bits = state_bits-stride
    assert path[0] >> stride == path[-1] & ((1 << low_bits)-1)
    assert np.all(path[1:] >> stride == path[:-1] & ((1 << low_bits)-1))
    return np.asarray(mapping)[path], path


def pack_path(path, state_bits=16, stride=2):
    """Pack a circular state path at exactly ``stride`` bits per output."""
    states=np.asarray(path,dtype=np.int64)
    mask=(1<<state_bits)-1
    if np.any(states<0) or np.any(states>mask):
        raise ValueError("state outside codebook")
    lowmask=(1<<(state_bits-stride))-1
    if not np.all(states[1:]>>stride==states[:-1]&lowmask):
        raise ValueError("invalid transition")
    if states[0]>>stride!=states[-1]&lowmask:
        raise ValueError("path is not circular")
    bits=[(int(states[0])>>shift)&1 for shift in range(state_bits-1,-1,-1)]
    for state in states[1:]:
        bits.extend((int(state)>>shift)&1 for shift in range(stride-1,-1,-1))
    bits=bits[:len(states)*stride]
    return np.packbits(np.asarray(bits,dtype=np.uint8),bitorder="big").tobytes()


def unpack_path(payload,length,state_bits=16,stride=2):
    bits=np.unpackbits(np.frombuffer(payload,dtype=np.uint8),bitorder="big")[:length*stride]
    if len(bits)!=length*stride or length*stride%8:
        raise ValueError("invalid packed length")
    bits=np.r_[bits,bits[:state_bits-stride]]
    first=int(sum(int(bit)<<(state_bits-1-i) for i,bit in enumerate(bits[:state_bits])))
    path=np.empty(length,dtype=np.int32)
    path[0]=first
    for i in range(1,length):
        new=int(sum(int(bit)<<(stride-1-j) for j,bit in enumerate(bits[state_bits+(i-1)*stride:state_bits+i*stride])))
        path[i]=((path[i-1]<<stride)&((1<<state_bits)-1))|new
    if not np.array_equal(pack_path(path,state_bits,stride),payload):
        raise AssertionError("packed trellis roundtrip mismatch")
    return path


def run(samples=256, out=Path("results/e04-07-reference-optimization/reference/qtip-gaussian-tailbite"),
        normalize=False, names=("1mad", "3inst")):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261127)
    targets = rng.standard_normal((samples,256))
    rows=[]; begin=time.perf_counter()
    for name, function in (("1mad",one_mad),("3inst",three_inst)):
        if name not in names:
            continue
        mapping=function(np.arange(65536)).astype(np.float32)
        codebook_rms=float(np.sqrt(np.mean(mapping.astype(np.float64)**2)))
        if normalize:
            mapping /= codebook_rms
        for index, target in enumerate(targets):
            decoded, path=quantize(target,mapping)
            rows.append(dict(codebook=name,sample=index,mse=float(np.mean((target-decoded)**2)),
                             rate_bpw=2.,dimension=256,state_bits=16,tail_biting=True,
                             state_first=int(path[0]),state_last=int(path[-1]),
                             codebook_rms=codebook_rms,normalized=normalize))
        values=np.array([r["mse"] for r in rows if r["codebook"]==name])
        print(name,float(values.mean()),float(1.96*values.std(ddof=1)/np.sqrt(len(values))),flush=True)
    with (out/"metrics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    (out/"config.json").write_text(json.dumps(dict(seed=20261127,samples=samples,
        distribution="IID standard Gaussian",dimension=256,state_bits=16,stride=2,
        normalize_codebook_rms=normalize,codebooks=names,
        input_cast="float16 before Viterbi, as pinned upstream",cost_dtype="float32",
        algorithm="pinned upstream two-pass midpoint overlap approximation",
        runtime_s=time.perf_counter()-begin,
        source_commit="e90c6688c8dfae326a3a81b5eb032db7c6680ec0",
        note="Independent algorithm recreation; no direct tensor parity against upstream CUDA implementation"),indent=2)+"\n")


if __name__ == "__main__":
    run()
