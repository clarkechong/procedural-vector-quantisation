"""Experiment 03: windowed-codebook search and fresh-row rate–distortion.

Run from the repository root with ``python -m experiments.synthesis.trellis_experiment``.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.encoding.codebooks import one_mad
from experiments.generator_algebra import bitvector
from experiments.manifests.weights import Weights
from experiments.synthesis import miniature
from experiments.synthesis.guided import parse_program
from experiments.synthesis.trellis import (decode_bits, dependency_graph, encode,
                                           linear_map, window_dependency)


LENGTHS = (4, 8, 16, 32, 64, 128, 256, 1024)
COUNTS = (64, 64, 64, 48, 32, 24, 16, 8)
OUT = Path("results/e03-trellis")


def sample_rows(checkpoint, prior_manifests, lengths=LENGTHS, counts=COUNTS):
    """Disjoint rows for every new length and split, absent from prior samples."""
    rng = np.random.default_rng(20261128)
    old = set()
    for manifest in prior_manifests:
        old.update((entry["layer"], entry["projection"], entry["row"])
                   for entry in manifest)
    weights = Weights(checkpoint)
    matrices = {(layer, "down_proj"): weights.get(layer, "down_proj")
                for layer in miniature.LAYERS}
    pools = {}
    for (layer, projection), matrix in matrices.items():
        fresh = rng.permutation([row for row in range(matrix.shape[0])
                                 if (layer, projection, row) not in old])
        pools[layer] = dict(zip(("calibration", "test"), np.array_split(fresh, 2)))
    result = {}
    manifest = []
    for length, count in zip(lengths, counts):
        result[length] = {}
        for split in ("calibration", "test"):
            blocks = []
            selections = {layer: rng.choice(pools[layer][split],
                            count // len(miniature.LAYERS) + 1, replace=False).tolist()
                          for layer in miniature.LAYERS}
            for i in range(count):
                layer = miniature.LAYERS[i % len(miniature.LAYERS)]
                row = selections[layer].pop()
                matrix = matrices[(layer, "down_proj")]
                start = int(rng.integers(0, matrix.shape[1] - length + 1))
                blocks.append(matrix[row, start:start + length])
                manifest.append(dict(length=length, split=split, layer=layer,
                                     projection="down_proj", row=int(row), start=start))
            result[length][split] = np.asarray(blocks, dtype=np.float64)
    return result, manifest


def fit_scale(rows, state_bits, stride, mapping, starts=(0.7, 1.0, 1.4), passes=2):
    """Alternating exact Viterbi assignments and shared least-squares scale."""
    rms = np.sqrt(np.mean(rows * rows))
    base = float(rms / max(np.sqrt(np.mean(mapping * mapping)), 1e-12))
    best = (np.inf, None)
    for multiplier in starts:
        scale = base * multiplier
        for _ in range(passes):
            numerator = denominator = 0.0
            for row in rows:
                reconstructed, _, _ = encode(row, state_bits, stride, mapping * scale)
                unscaled = reconstructed / scale
                numerator += float(row @ unscaled)
                denominator += float(unscaled @ unscaled)
            scale = max(numerator / denominator, 1e-12) if denominator else scale
        error = score(rows, state_bits, stride, mapping * scale)["sse"]
        if error < best[0]:
            best = (error, scale)
    return best[1]


def score(rows, state_bits, stride, values):
    error = 0.0
    for row in rows:
        _, _, loss = encode(row, state_bits, stride, values)
        error += loss
    energy = float(np.square(rows).sum())
    return {"sse": error, "nmse": error / energy, "rows": len(rows),
            "weights": int(rows.size)}


def cheap_multiplier_search(calibration):
    """Synthesize the byte multiplier for the original five-bit window family."""
    ranked = []
    for multiplier in range(8, 256, 16):  # all 8*odd mod 256
        mapping = linear_map(5, multiplier)
        scale = fit_scale(calibration, 5, 1, mapping)
        result = score(calibration, 5, 1, mapping * scale)
        ranked.append(dict(multiplier=multiplier, scale=scale,
                           calibration_nmse=result["nmse"]))
    return sorted(ranked, key=lambda item: item["calibration_nmse"])


def simplification_audit(guided):
    """Check all 256 original seeds and demonstrate literal constant folding."""
    original = parse_program(guided["winners"][0]["program"])
    folded = (bitvector.Instruction("SHR", ("s", "i")),
              bitvector.Instruction("MUL", ("r0", 120)))
    old_book = bitvector.codebook(original, 8, 4)
    new_book = bitvector.codebook(folded, 8, 4)
    assert np.array_equal(old_book, new_book)
    window_book = np.asarray([decode_bits(seed, 4, 5, 1, linear_map(5))
                              for seed in range(256)])
    assert np.array_equal(old_book, window_book)
    graph = dependency_graph(original, 8, 4)
    analytic = window_dependency(5, 1, 4, linear_map(5))
    assert np.array_equal(graph, analytic)
    return {"original": bitvector.expression(original),
            "folded": bitvector.expression(folded),
            "exact_codebook_equivalence": True,
            "distinct_codewords": int(len(np.unique(old_book, axis=0))),
            "dependency_matrix": graph.astype(int).tolist(),
            "bits_per_coordinate": graph.sum(axis=1).tolist(),
            "adjacent_shared_bits": [int(np.count_nonzero(graph[i] & graph[i+1]))
                                     for i in range(3)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/smollm2-135m.safetensors")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prior = Path("results/e01-miniature-codebook")
    followup = Path("results/e02-codebook-followup")
    guided = json.loads((followup / "guided.json").read_text())
    old_manifest = json.loads((prior / "sample_manifest.json").read_text())
    final_manifest = json.loads((followup / "final_manifest.json").read_text())
    samples, manifest = sample_rows(args.checkpoint, (old_manifest, final_manifest))
    (out / "sample_manifest.json").write_text(json.dumps(manifest) + "\n")
    audit = simplification_audit(guided)
    search = cheap_multiplier_search(samples[4]["calibration"])
    # The discovered mapping is frozen for the length sweep. The new multiplier
    # search remains a separate exploratory comparison on the same new samples.
    configurations = [("discovered_120", 5, 1, linear_map(5, 120)),
                      ("discovered_120_stride2", 5, 2, linear_map(5, 120)),
                      ("byte_120_L4", 4, 1, linear_map(4, 120)),
                      ("one_mad_L8_k1", 8, 1, one_mad(8)),
                      ("one_mad_L8_k2", 8, 2, one_mad(8)),
                      ("one_mad_L12_k1", 12, 1, one_mad(12)),
                      ("one_mad_L12_k2", 12, 2, one_mad(12)),
                      ("one_mad_L16_k1", 16, 1, one_mad(16)),
                      ("one_mad_L16_k2", 16, 2, one_mad(16))]
    records = []
    for length in LENGTHS:
        for name, bits, stride, mapping in configurations:
            # Bound the high-state comparisons while keeping a common subset
            # of the new rows for both discovered and QTIP-style mappings.
            cap = 4 if bits == 16 else 12 if bits == 12 else len(samples[length]["test"])
            cal = samples[length]["calibration"][:cap]
            test = samples[length]["test"][:cap]
            scale = fit_scale(cal, bits, stride, mapping)
            trained = score(cal, bits, stride, mapping * scale)
            measured = score(test, bits, stride, mapping * scale)
            record = dict(length=length, method=name, state_bits=bits, stride=stride,
                          payload_bits=bits + stride * (length - 1),
                          payload_bpw=(bits + stride * (length - 1)) / length,
                          scale=scale, calibration_nmse=trained["nmse"],
                          test_nmse=measured["nmse"], test_rows=len(test),
                          test_weights=int(test.size))
            records.append(record)
            print(f"{length:4d} {name:22s} {len(test):2d} rows "
                  f"{measured['nmse']:.6f} NMSE", flush=True)
    with Path(args.checkpoint).open("rb") as source:
        checkpoint_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    result = {"simplification": audit, "multiplier_search": search,
              "rate_distortion": records, "elapsed_seconds": time.monotonic()-started,
              "checkpoint_sha256": checkpoint_sha256,
              "sample": "new rows absent from both Experiment 01 and 02 samples; three down_proj matrices",
              "boundary": "open, first state_bits payload bits then stride bits per next weight",
              "scale_fit": "three initial scales and two Viterbi/least-squares passes; local optimization",
              "rate_excludes": "one shared global scale and mapping description; payload only"}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
