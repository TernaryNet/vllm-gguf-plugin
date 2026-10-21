# vLLM GGUF Quantization Plugin

This plugin provides out-of-tree GGUF quantization support for vLLM after
in-tree support deprecation
([vllm-project/vllm#39583](https://github.com/vllm-project/vllm/issues/39583)).

## Installation

### Prerequisites

- CUDA toolkit or ROCm toolkit

We recommend [uv](https://docs.astral.sh/uv/) for package management. If you
don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From Source

1. Clone this repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-gguf-plugin
   cd vllm-gguf-plugin
   ```

2. If vLLM is not already installed, install it first:

   ```bash
   uv pip install vllm --torch-backend=auto
   ```

3. Build and install the plugin against the PyTorch installation used by
   vLLM:

   ```bash
   uv pip install -e . --no-build-isolation
   ```

   Disabling build isolation ensures that the CUDA extension is compiled
   against the same PyTorch installation used by vLLM at runtime.

## Development

After completing the editable source installation above, install and run the
development tooling:

```bash
uv pip install -e .[dev] --torch-backend=auto
pre-commit install
pre-commit run --all-files
```

The same hooks also run in GitHub Actions on every push and pull request.

## Usage

```bash
vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --tokenizer Qwen/Qwen3-0.6B
```

Qwen 3.5 MTP speculative decoding loads the `nextn` block embedded in the same
GGUF; it does not download separate Hugging Face MTP weights:

```bash
vllm serve unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M \
  --tokenizer Qwen/Qwen3.5-4B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

For a GGUF without a `nextn` block, omit `--speculative-config`; the backbone
loads normally without MTP.

## PrismML Bonsai (PTQ1_0 ternary)

[PrismML Bonsai 2](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
checkpoints use the private `PTQ1_0` GGML tensor type (type 143: 128 ternary
weights per block packed base-3 with one fp16 scale, 1.75 bits per weight) and
store every low-bit matrix in a blockwise Hadamard-rotated basis declared by
`prism.hadamard.*` metadata. The plugin registers the type with the `gguf`
package, ships CUDA/Triton dequantize and GEMV kernels for it, and applies the
sign flip plus normalized Walsh-Hadamard rotation to activations before every
folded matmul (and the inverse after latent token-embedding lookups), exactly
as the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) does.

```bash
vllm serve prism-ml/Ternary-Bonsai-2-27B-gguf:PTQ1_0 --tokenizer Qwen/Qwen3.8-27B
vllm serve prism-ml/Ternary-Bonsai-2-2B-gguf:PTQ1_0 --tokenizer Qwen/Qwen3.5-2B
```

Notes:

- The activation rotation mixes blocks of `prism.hadamard.block_size` input
  features (1024 for the 27B model), so tensor parallelism must keep whole
  blocks on every rank: row-parallel inputs of 5120, 6144 and 17408 features
  limit the 27B model to `--tensor-parallel-size 1`.
- Only quantized (`PTQ1_0`) folded weights are supported; the `F16` and
  `PQ2_0` Bonsai files are rejected with an explicit error rather than loaded
  without their transform.
- Large batches fall back to dequantize + GEMM; the MMVQ path serves small
  decode batches.

## Tested model coverage

The plugin uses vLLM's model implementations and a generic GGUF weight
adapter, so model compatibility is broader than a fixed allowlist. The models
below are covered by the repository's generation tests and are the best-known
starting points:

| Modality | Model family | Tested GGUF quantization |
| --- | --- | --- |
| Text | Qwen 2.5 | Q6_K |
| Text | Qwen 3 | Q8_0 |
| Text | Phi 3.5 | IQ4_XS |
| Text | GPT-2 | Q4_K_M |
| Text | StableLM | Q4_K_M |
| Text | Gemma 3 | Q4_0 |
| Text | OLMoE | Q4_0 |
| Vision-language | Gemma 3 | Q4_0 backbone with F16 projector |
| Vision-language | Gemma 4 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.5 | Q4_K_M backbone with BF16 projector |
| Vision-language | Qwen 3.6 | UD-IQ2_XXS backbone with BF16 projector |
| Text | PrismML Bonsai 2 (Qwen 3.5/3.8 base) | PTQ1_0 with Hadamard-folded weights |
| Image generation | Z-Image-Turbo | Q4_0 |
| Image generation | FLUX.2-klein | Q8_0 |

Other vLLM-supported architectures may work when their GGUF tensor names map
to the corresponding Hugging Face model. A model appearing in vLLM's general
supported-model list does not by itself guarantee GGUF compatibility. When
reporting an unsupported model, include the model repository, quantization,
plugin and vLLM versions, and the complete weight-mapping error.
