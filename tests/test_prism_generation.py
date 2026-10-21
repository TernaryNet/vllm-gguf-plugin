# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end generation with a PrismML Bonsai PTQ1_0 GGUF.

Bonsai weights are ternary re-trainings of their base models, so unlike the
other GGUF generation tests there is no unquantized checkpoint whose logprobs
they should match. The test instead checks that the Hadamard-folded model
produces the expected greedy answers.
"""

import os

import pytest
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "true"

BONSAI_2B = ("prism-ml/Ternary-Bonsai-2-2B-gguf:PTQ1_0", "Qwen/Qwen3.5-2B")

PROMPTS_AND_EXPECTED = [
    ("The capital of France is", "Paris"),
    ("What is 17 * 23? Answer with just the number.", "391"),
]


@pytest.mark.slow
@pytest.mark.parametrize("model, tokenizer_name", [BONSAI_2B])
def test_bonsai_ptq1_0_generation(model: str, tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt, _ in PROMPTS_AND_EXPECTED
    ]
    llm = LLM(
        model=model,
        tokenizer=tokenizer_name,
        max_model_len=2048,
        gpu_memory_utilization=0.6,
    )
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48))
    for (prompt, expected), output in zip(PROMPTS_AND_EXPECTED, outputs):
        text = output.outputs[0].text
        assert expected in text, f"{prompt!r} -> {text!r}"
