# Computational codebooks and procedural weights

This repository studies compact programs and codebooks for pretrained neural-network weights. The [study index](docs/index.md) summarizes the evidence and links to reports and preserved artifacts. The [research proposal](docs/proposal.md) describes the broader questions.

## Layout

| Path | Purpose |
| --- | --- |
| [`docs/`](docs/) | Study reports, methods, status, and historical notes |
| [`experiments/`](experiments/) | Executable experiment and reporting code |
| [`results/`](results/) | Preserved measurements, configurations, and artifacts by study ID |
| [`scripts/`](scripts/) | Reproduction scripts that coordinate multiple experiment modules |
| [`tests/`](tests/) | Reference and codec tests |
| `data/`, `runs/` | Local inputs and scratch runs; ignored by Git |

## Quick start

Run commands from the repository root:

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests
uv run --locked python -m experiments.rate_distortion.pilot --out runs/synthetic --tiles 4 --repeats 2
```

The base environment covers CPU experiments. GPU studies require the relevant ROCm or CUDA environment and device access. Consult the [study index](docs/index.md) for input revisions, exact commands, and measurement limits. The [archived overview](docs/archive/overview-2026-11-27.md) retains the earlier detailed walkthrough.
