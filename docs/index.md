# Study index

Reports live in `docs/studies/`; preserved measurements live in matching directories under `results/`. Study IDs are stable. `s` identifies the early, unnumbered studies; `e` follows the numbered experiment series. See [result layout and provenance](methods/result-layout.md) before adding a run.

| Study | Report | Artifacts |
| --- | --- | --- |
| s01 · dictionary pursuit and ordering | [historical analysis](archive/results-index-2026-11-27.md) | [results](../results/s01-dictionary-pursuit/) |
| s02 · SeedLM reproduction | [report](studies/s02-seedlm-reproduction.md) | [results](../results/s02-seedlm-reproduction/) |
| s03 · computational codebooks | [historical analysis](archive/results-index-2026-11-27.md#computational-codebook-follow-up) | [results](../results/s03-computational-codebooks/) |
| s04 · generator search | [report](studies/s04-generator-search.md) | [results](../results/s04-generator-search/) |
| s05 · generator evaluation | [report](studies/s05-generator-evaluation.md) | [results](../results/s05-generator-evaluation/) |
| s06 · hardware evaluation | [laptop](studies/s06-laptop-rtx3050.md) · [MI300X](studies/s06-mi300x-3p5.md) | [results](../results/s06-hardware-evaluation/) |
| s07 · Llama-2 reference | [report](studies/s07-llama2-reference.md) | [results](../results/s07-llama2-reference/) |
| e01 · miniature codebook | [report](studies/e01-miniature-codebook.md) | [results](../results/e01-miniature-codebook/) |
| e02 · codebook follow-up | [report](studies/e02-codebook-followup.md) | [results](../results/e02-codebook-followup/) |
| e03 · trellis | [report](studies/e03-trellis.md) | [results](../results/e03-trellis/) |
| e04–07 · reference and optimization | [report](studies/e04-07-reference-optimization.md) | [results](../results/e04-07-reference-optimization/) |
| e08–12 · low-rate training | [report](studies/e08-12-low-rate-training.md) | [results](../results/e08-12-low-rate-training/) |
| e13–18 · activation and landscape | [interim report](studies/e13-18-activation-landscape.md) | [results](../results/e13-18-activation-landscape/) |
| e19–21 · QTIP and training maps | [interim report](studies/e19-21-qtip-training.md) | [results](../results/e19-21-qtip-training/) |
| e22–26 · E-means and ProSparse | [decision report](studies/e22-26-emeans-prosparse.md) | [results](../results/e22-26-emeans-prosparse/) |

## Current research position

The later [e22–26 screen](studies/e22-26-emeans-prosparse.md) found no useful exact-grouping gain at the tested geometries, so its expensive kernel and training branches were stopped. The [e19–21](studies/e19-21-qtip-training.md) and [e13–18](studies/e13-18-activation-landscape.md) reports remain interim. See each report for its specific completed gates and limits. The [dated status snapshot](archive/status-2026-11-27.md) is retained for historical context.

## Methods

- [Provenance and input sampling](methods/provenance.md)
- [Result layout and run manifests](methods/result-layout.md)
- [Reference implementations](methods/reference-methods.md)
- [Llama-2 evaluation](methods/llama2-7b-evaluation.md)
- [Training](methods/training.md)
- [Target geometry](methods/target-geometry.md)
