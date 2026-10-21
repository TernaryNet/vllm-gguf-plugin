"""Closed-loop concurrency benchmark for llama-server.

N concurrent users, ~P-token prompts, G forced output tokens each.
"""

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request

PORT = 8099
BASE = [
    (
        "You are given a long transcript of a planning meeting between the "
        "engineering, product and  operations teams of a mid-sized logistics company. "
        "The teams discuss the migration of their  warehouse management system to a "
        "new event-driven architecture, the risks involved in cutting  over during "
        "the holiday season, the staffing plan for the on-call rotation, the budget "
        "for  additional cloud capacity, and the metrics that will be used to judge "
        "whether the rollout  succeeded. Several participants disagree about whether "
        "to run the old and new systems in  parallel for a month or to switch over in "
        "a single weekend. Please write a detailed summary  of the discussion, list "
        "every decision that was reached, list every open question, and then  propose "
        "a concrete week-by-week rollout plan with owners and rollback criteria. Be "
        "thorough  and explain the reasoning behind each recommendation."
    ),
]


def make_prompts(tokenizer_name, n, target_tokens):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    prompts = []
    for i in range(n):
        text = f"Request {i + 1}: " + BASE[0]
        ids = tok(text)["input_ids"]
        while len(ids) < target_tokens:
            ids = (
                ids
                + tok(" Provide additional detail and context in your answer.")[
                    "input_ids"
                ]
            )
        prompts.append(tok.decode(ids[:target_tokens]))
    return prompts


def start_server(server_bin, gguf, n_parallel, ctx_per_slot, log):
    cmd = [
        server_bin,
        "-m",
        gguf,
        "-ngl",
        "99",
        "-fa",
        "on",
        "-np",
        str(n_parallel),
        "-c",
        str(n_parallel * ctx_per_slot),
        "--port",
        str(PORT),
        "--host",
        "127.0.0.1",
        "--no-warmup",
        "-t",
        "4",
        "-b",
        "2048",
        "-ub",
        "512",
    ]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("llama-server exited early") from None
            time.sleep(1)
    raise RuntimeError("llama-server did not become healthy")


def stream_request(prompt, n_predict, result, idx, barrier):
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    t0 = time.perf_counter()
    t_first = None
    n_tokens = 0
    prompt_n = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("content") and t_first is None:
                t_first = time.perf_counter()
            if "stop" in chunk and chunk["stop"]:
                n_tokens = chunk.get("tokens_predicted", n_tokens)
                prompt_n = chunk.get("tokens_evaluated")
            elif chunk.get("content") is not None:
                n_tokens += 1
    t_end = time.perf_counter()
    result[idx] = {
        "ttft": (t_first or t_end) - t0,
        "total": t_end - t0,
        "tokens": n_tokens,
        "prompt_tokens": prompt_n,
        "t0": t0,
        "t_end": t_end,
    }


def run_users(prompts, n_predict):
    results = [None] * len(prompts)
    barrier = threading.Barrier(len(prompts))
    threads = [
        threading.Thread(
            target=stream_request, args=(p, n_predict, results, i, barrier)
        )
        for i, p in enumerate(prompts)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = max(r["t_end"] for r in results) - min(r["t0"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    per_user = [
        r["tokens"] / (r["total"] - r["ttft"]) for r in results
    ]  # decode rate after first token
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
        "prompt_tokens": results[0]["prompt_tokens"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument(
        "--server",
        default=os.path.expanduser("~/work/llama.cpp/build/bin/llama-server"),
    )
    ap.add_argument("--users", type=int, nargs="*", default=[1, 8, 16, 32, 48])
    ap.add_argument("--prompt-tokens", type=int, default=175)
    ap.add_argument("--output-tokens", type=int, default=1024)
    ap.add_argument("--ctx-per-slot", type=int, default=1536)
    ap.add_argument("--out", default="bench_llamacpp_users.json")
    args = ap.parse_args()

    all_results = []
    for n in args.users:
        prompts = make_prompts(args.tokenizer, n, args.prompt_tokens)
        with open(f"/tmp/llama_server_np{n}.log", "w") as log:
            proc = start_server(args.server, args.gguf, n, args.ctx_per_slot, log)
            try:
                run_users(prompts[:1], 16)  # warm up kernels / graphs
                res = run_users(prompts, args.output_tokens)
            finally:
                proc.terminate()
                proc.wait(timeout=60)
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
