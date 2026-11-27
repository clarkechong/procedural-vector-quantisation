"""Open-boundary Gaussian component check; not QTIP Table 1 reproduction."""

import csv
import json
from pathlib import Path
import time

import numpy as np

from experiments.reference.qtip.codebooks import one_mad, three_inst
from experiments.synthesis.trellis import encode


def main():
    out = Path("results/e04-07-reference-optimization/reference/qtip-gaussian-pilot")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261127)
    samples = rng.standard_normal((256, 256))
    rows = []
    start = time.perf_counter()
    states = np.arange(65536)
    for name, function in (("1mad", one_mad), ("3inst", three_inst)):
        mapping = function(states).astype(np.float64)
        for number, target in enumerate(samples):
            decoded, _, _ = encode(target, 16, 2, mapping)
            rows.append(dict(codebook=name, sample=number, mse=float(np.mean((target-decoded)**2)),
                             boundary="open", length=256, state_bits=16, stride=2))
    with (out/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out/"config.json").write_text(json.dumps(dict(seed=20261127, samples=256,
             distribution="IID standard normal", boundary="open", scale="fixed upstream normalization",
             runtime_s=time.perf_counter()-start,
             note="Small component pilot; paper Table 1 uses its own boundary and scale conventions"), indent=2)+"\n")
    for name in ("1mad", "3inst"):
        values=np.array([r["mse"] for r in rows if r["codebook"] == name])
        print(name, "mean", values.mean(), "95% normal CI", (values.mean()-1.96*values.std(ddof=1)/np.sqrt(len(values)),
              values.mean()+1.96*values.std(ddof=1)/np.sqrt(len(values))))


if __name__ == "__main__":
    main()
