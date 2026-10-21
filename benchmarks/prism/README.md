# PrismML Bonsai benchmarks

Scripts used to validate and measure PTQ1_0 support against the
[PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (built with
`cmake -B build -DGGML_CUDA=ON` and `llama-server` in `build/bin`).

- `bench_ptq1_0.py` times the MMVQ, Triton GEMM and dequantize + GEMM paths on
  the Bonsai-27B layer shapes for a range of batch sizes; the dispatch
  thresholds in `quantization/linear.py` come from its output.
- `compare_llamacpp.py` runs the same raw prompts through `llama-server` and
  vLLM and prints both greedy continuations with first-token top-5 logprobs.
- `bench_llamacpp_users.py` is a closed-loop load test against `llama-server`:
  N concurrent users, ~175-token prompts, 1024 forced output tokens; reports
  aggregate tok/s, per-user generation rate and time to first token.

```bash
GGUF=$(ls ~/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/*/Ternary-Bonsai-2-27B-PTQ1_0.gguf)
python benchmarks/prism/compare_llamacpp.py --gguf "$GGUF" \
    --model prism-ml/Ternary-Bonsai-2-27B-gguf:PTQ1_0 --tokenizer Qwen/Qwen3.8-27B
python benchmarks/prism/bench_llamacpp_users.py --gguf "$GGUF" --tokenizer Qwen/Qwen3.8-27B
```
