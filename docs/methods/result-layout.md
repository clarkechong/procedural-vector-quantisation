# Result layout and provenance

Use `results/<study-id>/<run-id>/` for new preserved runs. Study IDs appear in the [study index](../index.md). Run IDs use lowercase kebab-case and describe the model, method, budget, or seed as needed. Keep temporary output under ignored `runs/` until it has been checked.

A new run should contain `manifest.json`, its raw measurements, and any packed payloads or plots needed to inspect or reproduce the result. The manifest should record the source commit, exact command, model and dataset revisions, input checksums, parameters and random seeds, hardware and software environment, outcome status, and SHA-256 hashes of preserved artifacts. Mark unavailable fields explicitly; do not infer them from a directory name.

The migrated historical runs retain their original configuration and manifest schemas. Some historical JSON files include their original `experiments/results/...` paths as provenance. Those fields describe where the run was created, while current reproduction commands and documentation use `results/...`. Existing raw measurements and binary artifacts were moved without changing their contents.

Keep reports and interpretation in `docs/studies/`. Put plotting and summarizing programs in `experiments/reporting/`, and coordinating shell scripts in `scripts/`. Do not place executable code or prose reports among raw results.
