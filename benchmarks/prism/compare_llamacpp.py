"""Compare llama.cpp (PrismML fork, CUDA) and vLLM greedy continuations on raw prompts.

Starts llama-server on the GGUF, queries /completion with n_probs, then runs vLLM
on the same prompts and prints both continuations and first-token top-5 logprobs.
"""

import argparse
import json
import math
import os
import subprocess
import time
import urllib.request


def llamacpp_run(server_bin, gguf, prompts, max_tokens, port=8099):
    log = open("/tmp/llama_server.log", "w")  # noqa: SIM115 - outlives the call
    proc = subprocess.Popen(
        [
            server_bin,
            "-m",
            gguf,
            "-ngl",
            "99",
            "-fa",
            "on",
            "-c",
            "4096",
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "-np",
            "1",
            "--no-warmup",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ).read()
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not start; see /tmp/llama_server.log")
        results = []
        for p in prompts:
            body = json.dumps(
                {
                    "prompt": p,
                    "n_predict": max_tokens,
                    "temperature": 0.0,
                    "top_k": 1,
                    "n_probs": 5,
                    "samplers": ["top_k"],
                    "cache_prompt": False,
                }
            ).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/completion",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            first = r.get("completion_probabilities", [{}])[0]
            cands = first.get("top_logprobs") or first.get("probs") or []
            top = []
            for c in cands[:5]:
                tok = c.get("token", c.get("tok_str", ""))
                lp = c.get("logprob", None)
                if lp is None and "prob" in c:
                    lp = math.log(max(c["prob"], 1e-30))
                top.append((round(lp, 4) if lp is not None else None, tok))
            results.append(
                {
                    "prompt": p,
                    "text": r["content"],
                    "first_top5": top,
                    "tokens_predicted": r.get("tokens_predicted"),
                    "timings": {
                        k: r.get("timings", {}).get(k)
                        for k in ("predicted_per_second", "prompt_per_second")
                    },
                }
            )
        return results
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def vllm_run(model, tokenizer, prompts, max_tokens):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        max_num_seqs=4,
    )
    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5)
    )
    res = []
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        first = c.logprobs[0] if c.logprobs else {}
        top = sorted(
            ((round(lp.logprob, 4), lp.decoded_token) for tid, lp in first.items()),
            reverse=True,
        )[:5]
        res.append({"prompt": p, "text": c.text, "first_top5": top})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument(
        "--server",
        default=os.path.expanduser("~/work/llama.cpp/build/bin/llama-server"),
    )
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--skip-llamacpp", action="store_true")
    ap.add_argument("--out", default="compare.json")
    ap.add_argument(
        "--prompts",
        nargs="*",
        default=[
            "The capital of France is",
            "Once upon a time, in a small village by the sea,",
            'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
            "Q: What is 17 * 23?\nA:",
        ],
    )
    args = ap.parse_args()
    out = {}
    if not args.skip_llamacpp:
        out["llamacpp"] = llamacpp_run(
            args.server, args.gguf, args.prompts, args.max_tokens
        )
        print(json.dumps(out["llamacpp"], indent=1), flush=True)
    if not args.skip_vllm:
        out["vllm"] = vllm_run(
            args.model, args.tokenizer, args.prompts, args.max_tokens
        )
        print(json.dumps(out["vllm"], indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    if "llamacpp" in out and "vllm" in out:
        for a, b in zip(out["llamacpp"], out["vllm"]):
            same = a["text"].strip() == b["text"].strip()
            print("=" * 80)
            print("PROMPT:", repr(a["prompt"]))
            print("llama.cpp:", repr(a["text"]))
            print("vllm     :", repr(b["text"]))
            print("identical greedy text:", same)
            print("llama.cpp top5:", a["first_top5"])
            print("vllm      top5:", b["first_top5"])


if __name__ == "__main__":
    main()
