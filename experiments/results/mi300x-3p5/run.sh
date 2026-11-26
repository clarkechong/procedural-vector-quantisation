#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=runs/mi300x/torchinductor
export TRITON_CACHE_DIR=runs/mi300x/triton
result=experiments/results/mi300x-3p5
output=runs/mi300x/synthesized_3p5
mkdir -p "$result" "$output"
trap 'printf "failed\n" > "$result/status.txt"' ERR
printf 'running\n' > "$result/status.txt"
git rev-parse HEAD > "$result/source_commit.txt"
git status --short > "$result/source_status.txt"
.venv/bin/python - <<'PY' > "$result/environment.txt"
import platform, torch, numpy, scipy, transformers, accelerate, pyarrow, tokenizers
print('python:', platform.python_version())
for name, module in [('torch', torch), ('numpy', numpy), ('scipy', scipy),
                     ('transformers', transformers), ('accelerate', accelerate),
                     ('pyarrow', pyarrow), ('tokenizers', tokenizers)]:
    print(name + ':', module.__version__)
print('hip:', torch.version.hip)
print('device:', torch.cuda.get_device_name(0))
print('properties:', torch.cuda.get_device_properties(0))
PY
.venv/bin/python -m experiments.evaluation.generator_model \
  --checkpoint data/smollm2-135m.safetensors --data data/smollm2 \
  --out "$output" --methods synthesized_3p5 \
  --encoder-backend rocm --score-backend compiled \
  --seed-tile 65536 --target-tile 32768 \
  --device cuda --dtype float32 --threads 6 \
  --context 1024 --batch 1 --calibration-tokens 512
mkdir -p "$result/synthesized_3p5"
cp "$output"/*.csv "$output"/*.json "$result/synthesized_3p5/"
MPLCONFIGDIR=runs/mi300x/matplotlib .venv/bin/python -m experiments.evaluation.plot_model_quality \
  --runs experiments/results/laptop-rtx3050 --mi300x "$result" --out "$result/plots"
printf 'complete\n' > "$result/status.txt"
