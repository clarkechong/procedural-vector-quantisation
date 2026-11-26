# Experiment inputs and provenance

`weights.py` reads local checkpoints and maps attention and MLP projection names to checkpoint tensors. `sampling.py` reproduces the paired screen and holdout row spans used by the SeedLM, codebook, and synthesis studies.

Store local checkpoints and datasets under `data/`. Keep each published configuration and sampling manifest alongside its metrics in `experiments/results/<run-id>/`. Historical manifests record the original run paths. Reproduction commands use paths in the current checkout.

For new runs, record the checkpoint revision and checksum, selected matrices, controls, tile layout, RNG seed, row and column selections, and split membership. Record the generator and constants, encoder settings, packed field widths, shared bytes, and hardware and software versions where applicable. Keep timing and distortion measurements with that configuration.
