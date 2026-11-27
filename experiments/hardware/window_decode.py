"""MI300X fused bitstream decode plus matrix-vector product microbenchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from experiments.synthesis.trellis import decode_bits, linear_map


@triton.jit
def window_matvec(payload, activation, output, scale, N: tl.constexpr,
                  BYTES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    valid = i < N
    byte = i // 8
    shift = i % 8
    first = tl.load(payload + row * BYTES + byte, mask=valid, other=0).to(tl.uint32)
    second = tl.load(payload + row * BYTES + byte + 1,
                     mask=valid & (byte + 1 < BYTES), other=0).to(tl.uint32)
    state = ((first | (second << 8)) >> shift) & 31
    raw = (state * 120) & 255
    signed = tl.where(raw >= 128, raw.to(tl.int32) - 256, raw.to(tl.int32))
    x = tl.load(activation + i, mask=valid, other=0)
    value = tl.sum(signed.to(tl.float32) * x * scale, 0)
    tl.store(output + row, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/results/trellis-followup/fused_benchmark.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("A ROCm/CUDA GPU is required")
    length, rows = args.length, args.rows
    width = (length + 4 + 7) // 8
    rng = np.random.default_rng(417)
    data = rng.integers(0, 256, (rows, width), dtype=np.uint8)
    last_bits = (length + 4) % 8
    if last_bits:
        data[:, -1] &= (1 << last_bits) - 1
    activation = rng.normal(size=length).astype(np.float32)
    packed = torch.from_numpy(data).to("cuda")
    vector = torch.from_numpy(activation).to("cuda")
    output = torch.empty(rows, device="cuda", dtype=torch.float32)
    scale = 0.0038
    launch = lambda: window_matvec[(rows,)](packed, vector, output, scale,
                                             length, width, triton.next_power_of_2(length))
    launch()
    torch.cuda.synchronize()
    mapping = linear_map(5) * scale
    expected = np.asarray([decode_bits(int.from_bytes(data[row], "little"), length, 5, 1,
                                       mapping) @ activation for row in range(min(rows, 16))])
    observed = output[:len(expected)].cpu().numpy()
    np.testing.assert_allclose(observed, expected, atol=2e-4, rtol=2e-5)
    milliseconds = triton.testing.do_bench(launch)
    # Same decoded values, stored densely in FP16 for a bandwidth-oriented
    # PyTorch matrix-vector reference. This is a separate storage format.
    indices = np.arange(length)
    low = data[:, indices // 8].astype(np.uint16)
    high = np.zeros_like(low)
    available = indices // 8 + 1 < width
    high[:, available] = data[:, indices[available] // 8 + 1]
    states = ((low | (high << 8)) >> (indices % 8)) & 31
    dense = torch.from_numpy((mapping[states]).astype(np.float16)).to("cuda")
    dense_vector = vector.to(torch.float16)
    dense_launch = lambda: torch.mv(dense, dense_vector)
    dense_launch()
    dense_ms = triton.testing.do_bench(dense_launch)
    properties = torch.cuda.get_device_properties(0)
    result = dict(gpu=torch.cuda.get_device_name(), gpu_arch=properties.gcnArchName,
                  rows=rows, weights_per_row=length,
                  payload_bytes=int(data.size), payload_bpw=8 * data.size / (rows * length),
                  milliseconds=milliseconds, decoded_weights_per_second=rows * length * 1000 / milliseconds,
                  dense_fp16_matvec_milliseconds=dense_ms,
                  dense_fp16_weight_bytes=int(dense.numel() * dense.element_size()),
                  max_abs_reference_difference=float(np.max(np.abs(observed - expected))),
                  reference_rows=len(expected), kernel="Triton fused decode plus one dot product per row",
                  note="Isolated microbenchmark; excludes encoder, model execution, and activation traffic from other layers")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
