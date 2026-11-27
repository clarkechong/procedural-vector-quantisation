# Experiments 22–26: bounded E-means and ProSparse screen

27 November 2026. This is a **negative offline screening result**, not a measured
MI300X speed result. Commands below ran in the repository `.venv`; GPU capture
parity used the authorized host device path. Raw files are adjacent to this report.

## Provenance and contract

- ProSparse source: `/root/work/sparse-2-4` at `96bc7dba211a7bc7b15a21de2a52d2c189e2144b`.
  Checkpoint `SparseLLM/prosparse-llama-2-7b` revision
  `5197d09bdea44d572a69330bf12db47f2c90e061`, layer 15, hidden 4096,
  intermediate 11008, exported FP16 weights. C4 revision
  `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`; 1000-document local sample
  SHA256 `f3f3c68830a1ec1ac56fd9047ebe3e82aa1113bcad8643ff74ec034df4e46166`.
  Source manifest and `reference_ffn.py` govern the exact function.
- Faiss E-means official source cloned to `/tmp/faiss-emeans` at
  `b14b3d693161c7b1cbc5f2e398c8234b42b04c8b`. Imported its unmodified
  `e_means_torch.Emeans` CPU backend directly. No Faiss binary or ROCm package
  was installed. This is **official component execution**, not a paper benchmark
  reproduction or GPU speedup.
- Grouping uses `data/activations/doc_*.npy` and corresponding capture metadata.
  Documents 64–75 were already exposed in the historical benchmark package;
  only 8 supplied full 128-row windows, yielding 18 windows. They are
  **regression data, not fresh held-out captures**. The checkpoint needed to
  generate new model activations was not present locally. No new document
  checkpoint download or multi-gigabyte capture run was made after this
  decisive negative screen. `captures_manifest.json` records file hashes,
  document IDs, layer, token counts, threshold source and model revision.
- Prefill windows stay within original document order and never pool future
  autoregressive decode steps. Full-window rearrangement is an optimistic
  offline upper-bound schedule. Grouping assumes exact masks are available
  after gate/up projection; it cannot save that upstream work.

## Gate 22: reference and FFN baseline — PASS (component level)

Official E-means on an imbalanced synthetic Gaussian mixture, N=4096, D=32,
K=32, three passes, identical random initialization: FP64 brute-force
nearest-centroid labels agreed 100% with official assignment. Held-out squared
distance was 15.782 for official E-means versus 49.228 for a basic three-pass
Lloyd control. CPU fit times were 0.091 s and 0.028 s respectively. This is a
small **adapted synthetic test**, not the paper's published embedding recipe;
the Lloyd control uses random initialization and no dead-cluster revival.
Official recipe, finalization, batch=256, decay schedule and revival policy
are recorded in `reference_emeans.json`.

The local FFN is `h = FATReLU(x W_gate^T, threshold=0.01) * (x W_up^T)` with
`FATReLU(g)=g` when `g >= 0.01`, zero otherwise; `y=h W_down^T`. The threshold
is on gate **before** multiplication. On four real captured rows (doc 67), GPU
recomputed `h` matched stored derived activation exactly. Dense down projection
and the same input multiplied by its exact-nonzero mask were bitwise equal.
The recomputed `y` differed from the saved remote-code output by at most
`6.103515625e-5` from FP16 reduction behavior. The CPU path did not exactly
match the GPU-derived `h`, so exact capture parity is asserted on GPU only.

## Gate 23: exact offline grouping — FAIL at screened geometries

Each group has exactly `g` rows. The reported `F_tile` is the count of occupied
fixed contiguous `b`-channel blocks across groups. Issued row-channel slots
are `F_tile * g * b`; `F_channel` and useful-slot fraction are in the raw CSV.
Methods: original order, random order, Hamming nearest-neighbor chain, greedy
union growth, fractional-centroid Lloyd with exact-capacity greedy assignment,
and exact legal pair swaps from the best starting schedule. The swap code
recomputes both groups after each proposed swap and accepts only decreasing
cost. Small all-zero, all-active, disjoint, and overlapping fixtures were
checked against an enumerated exact partition oracle. The lower bound is
`sum_block ceil(tokens_touching_block / g)` and is valid but can be loose.

| Rows/group × block width | Original tiles | Best refined tiles | Reduction | 95% document bootstrap | Absolute maximum reduction allowed by lower bound |
|---|---:|---:|---:|---:|---:|
| 4 × 8 | 776,674 | 760,351 | 2.10% | 1.43–2.68% | 16.71% |
| 8 × 8 | 395,107 | 390,814 | 1.09% | 0.59–1.66% | 16.59% |
| 4 × 16 | 395,835 | 392,746 | 0.78% | 0.37–1.20% | 4.59% |
| 8 × 16 | 198,132 | 197,583 | 0.28% | 0.11–0.49% | 3.41% |
| 16 × 16 | 99,072 | 99,021 | 0.05% | 0.03–0.08% | 1.58% |
| 8 × 32 | 99,072 | 99,021 | 0.05% | 0.01–0.12% | 0.28% |
| 16 × 32 | 49,536 | 49,533 | 0.006% | 0–0.016% | 0.032% |

All rows aggregate the same 18 document-local windows. Resampling is by
document (8 independent document IDs), not token. The local search is not a
global oracle. For 16- and 32-wide blocks the mathematical lower bound itself
rules out the proposed 15% tile reduction. For 8-wide blocks a larger global
gain is not mathematically excluded, but the strongest tested practical
schedule recovered only 1–2% and would incur grouping/packing overhead.
Individual-token channel density in representative documents is ~19–24%,
while 16-wide block occupancy is ~94–98% and 32-wide occupancy ~99–100%.
This saturation explains the weak regrouping opportunity.

Official E-means was also fit to five 128-row, 16-wide block-mask windows
with K=16 and three passes. Before exact-capacity repair, the largest cluster
held 58–84 rows against a capacity of 8. After the same balancing repair used
for Lloyd, E-means issued 11007–11008 tiles per window, versus ~11008 in capture
order. Thus lower centroid loss would not imply a useful hardware schedule.
These five CPU fits took 0.035–0.161 s each; no GPU grouping time was measured.
See `emeans_masks.csv`.

The proposed gate required ≥15% tile reduction against a strong practical
baseline **on fresh windows** or a credible measured ≥10% layer opportunity.
Neither emerged in the historical screening. Fresh confirmation was not run,
so the precise statement is **screening FAIL; fresh-window confirmation NOT RUN**.
The historical data provide a strong bound for b≥16, independent of grouping
algorithm. No claimed MFMA utilization or end-to-end latency follows.

## Gate 24 — stopped by Gate 23

The full-path MI300X sparse FFN benchmark, online grouping, profiler counters,
and causal decode replay were **NOT RUN**. Optimizing a grouped kernel for
these saturated block schedules is not justified by the screened headroom.

## Gate 25A: cheap prototype-mask oracle — FAIL at tested budgets

On training documents 0–31, fit 16 fixed prototypes to 1,014 token block masks,
each retaining the top 25% or 50% of 16-wide blocks by intermediate activation
energy. On 64 tokens from disjoint historical benchmark documents, select the
best prototype using the **actual intermediate and layer output**. This is an
offline oracle unavailable to a router based on FFN input. Mean per-token
output NMSE is the squared output-error norm divided by squared dense-output
norm. The per-token top-block control uses the same block budget; it too is
an oracle because it sees the intermediate.

| Kept blocks | 16-prototype output NMSE | Per-token top-block NMSE | One fixed mask NMSE |
|---:|---:|---:|---:|
| 25% | 0.3451 | 0.1010 | 0.4909 |
| 50% | 0.1819 | 0.0256 | 0.2981 |

At these budgets the small prototype set is far from the per-token oracle,
even before routing mistakes. One seed, one layer, two budgets and historical
documents are insufficient for a universal impossibility claim. No NLL or
multi-layer quality was measured; therefore this is a **layer-output screening
FAIL**, not the proposed ΔNLL gate. No learned-router or ProSparse fine-tuning
was launched. `prototype_oracle_q172.json` and `_q344.json` hold the results.

## Gate 26: independent bounded VQ diagnostic — component only

On ProSparse layer-15 `down_proj`, a shared d=4, K=256 codebook was trained
from 16,384 randomly selected tiles of output rows 0–2047 and evaluated on
4,096 disjoint tiles from rows 2048–4095. A common RMS scale and same random
initial centroids were used. Official E-means (three passes) gave held-out
weight NMSE **0.10187** in **0.359 s** CPU fitting; basic Lloyd (three passes)
gave **0.10350** in **0.082 s**. Both occupied 255/256 codes on test tiles.
Rounding the learned table to FP16 changed NMSE by <0.000002. The nominal
index rate is 2 bpw plus a 2,048-byte shared FP16 table, before scales,
packing, and other metadata. This is a sampled weight-distortion comparison,
not a full-matrix encoder, output-error result, or matched-storage frontier.
No LUT GEMM or direct-decode MI300X kernel was run, so the latency gate is
**NOT RUN**. Faster fit is not inferred; E-means was ~4.4× slower here.

## Decision matrix

| Question | Decision |
|---|---|
| Does E-means improve clustering optimization? | Synthetic component PASS for loss, not fit time; mask hardware objective negative after balancing. |
| Does exact grouping create useful fixed-block structure? | STOP for screened ProSparse layer-15 prefill geometries. Strong b≥16 lower bounds; b=8 practical gain ≤2.1%. |
| Does grouped MI300X execution win? | NOT RUN; offline gate failed. |
| Do learned masks unlock savings? | STOP this 16-prototype top-block design at 25–50% budget; no router or task-quality evidence. |
| Does codeword dot reuse beat decoding? | NOT RUN; CPU codebook diagnostic only. |

## Reproduction

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/faiss.git /tmp/faiss-emeans
git -C /tmp/faiss-emeans checkout b14b3d693161c7b1cbc5f2e398c8234b42b04c8b
git -C /tmp/faiss-emeans sparse-checkout set contrib/e_means benchs/e_means
.venv/bin/python experiments/clustering/reference_emeans.py
.venv/bin/python experiments/clustering/grouping_screen.py
.venv/bin/python experiments/clustering/grouping_screen.py --groups 4 8 --blocks 8 16 --output experiments/results/experiments-22-26/small_tiles
.venv/bin/python experiments/clustering/emeans_masks.py
.venv/bin/python experiments/clustering/prototype_oracle.py --q 172
.venv/bin/python experiments/clustering/prototype_oracle.py --q 344
.venv/bin/python experiments/clustering/vq_diagnostic.py
.venv/bin/python -m unittest tests.test_grouping_screen
```

No new model checkpoint, trained weights or production kernel is emitted.
