"""Closed-loop concurrency benchmark against a running vLLM OpenAI server.

Same load profile as ``bench_llamacpp_users.py``: N concurrent users, ~P-token
prompts, G forced output tokens each, one request per user. Start the server
first, e.g.::

    vllm serve prism-ml/Ternary-Bonsai-2-27B-gguf:PTQ1_0 --tokenizer Qwen/Qwen3.8-27B \\
        --max-num-seqs 64 --max-model-len 4096 --port 8000
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request

from bench_llamacpp_users import make_prompts


def stream_request(url, model, prompt, n_predict, result, idx, barrier):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": n_predict,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/completions", data=body, headers={"Content-Type": "application/json"}
    )
    barrier.wait()
    t0 = time.perf_counter()
    t_first = None
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            text = chunk["choices"][0].get("text", "")
            if text and t_first is None:
                t_first = time.perf_counter()
            if text:
                n_tokens += 1  # one token per streamed chunk
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                n_tokens = usage["completion_tokens"]
    t_end = time.perf_counter()
    result[idx] = {
        "ttft": (t_first or t_end) - t0,
        "total": t_end - t0,
        "tokens": n_tokens,
        "t0": t0,
        "t_end": t_end,
    }


def run_users(url, model, prompts, n_predict):
    results = [None] * len(prompts)
    barrier = threading.Barrier(len(prompts))
    threads = [
        threading.Thread(
            target=stream_request, args=(url, model, p, n_predict, results, i, barrier)
        )
        for i, p in enumerate(prompts)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = max(r["t_end"] for r in results) - min(r["t0"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    per_user = [r["tokens"] / (r["total"] - r["ttft"]) for r in results]
    per_user_e2e = [r["tokens"] / r["total"] for r in results]
    return {
        "users": len(prompts),
        "wall_s": wall,
        "total_tokens": total_tokens,
        "total_tok_s": total_tokens / wall,
        "per_user_decode_tok_s_median": statistics.median(per_user),
        "per_user_e2e_tok_s_median": statistics.median(per_user_e2e),
        "ttft_median_s": statistics.median(r["ttft"] for r in results),
        "ttft_max_s": max(r["ttft"] for r in results),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="prism-ml/Ternary-Bonsai-2-27B-gguf:PTQ1_0")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--users", type=int, nargs="*", default=[1, 8, 16, 32, 48])
    ap.add_argument("--prompt-tokens", type=int, default=175)
    ap.add_argument("--output-tokens", type=int, default=1024)
    ap.add_argument("--out", default="bench_vllm_users.json")
    args = ap.parse_args()

    all_results = []
    for n in args.users:
        prompts = make_prompts(args.tokenizer, n, args.prompt_tokens)
        run_users(args.url, args.model, prompts[:1], 16)  # warm up
        res = run_users(args.url, args.model, prompts, args.output_tokens)
        print(json.dumps(res), flush=True)
        all_results.append(res)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=1)

    header = (
        "Users | Total tok/s | Per user tok/s (decode) | Per user tok/s (e2e) | "
        "TTFT median | TTFT max"
    )
    print("\n" + header)
    for r in all_results:
        print(
            f"{r['users']:5d} | {r['total_tok_s']:11.1f} | "
            f"{r['per_user_decode_tok_s_median']:23.1f} | "
            f"{r['per_user_e2e_tok_s_median']:20.1f} | "
            f"{r['ttft_median_s']:9.2f} s | {r['ttft_max_s']:6.2f} s"
        )


if __name__ == "__main__":
    main()
