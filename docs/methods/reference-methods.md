# Reference source pins

The upstream repositories are kept outside this repository. Recreate the source checkouts used for parity tests:

```sh
git clone https://github.com/Cornell-RelaxML/qtip.git /tmp/qtip-official
git -C /tmp/qtip-official checkout e90c6688c8dfae326a3a81b5eb032db7c6680ec0
git clone https://github.com/mit-han-lab/smoothquant.git /tmp/smoothquant-official
git -C /tmp/smoothquant-official checkout c61476d728e42ae0d8a35e7e78494edcac3237b5
```

`tests/test_reference_parity.py` extracts only the upstream QTIP decode functions through Python AST so the all-state comparison can run without CUDA kernels. It imports upstream SmoothQuant directly for the shared-input fixture. The local NumPy and PyTorch ports are in `qtip/codebooks.py` and `smoothquant/transform.py`; upstream files are unmodified. Source SHA-256 values and software versions are recorded in `results/e04-07-reference-optimization/environment.json`.

The full official Llama-2 result is blocked in this session: no Hugging Face token is available, and a request explicitly made with `token=False` for `meta-llama/Llama-2-7b-hf` returned 401. This does not establish whether the user's Hugging Face account has access. QTIP model import also requires its CUDA `qtip_kernels` package and `glog`, absent in this MI300X environment. The public `relaxml/Llama-2-7b-QTIP-2Bit` config alone does not validate its quantization or model perplexity.
