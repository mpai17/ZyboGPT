"""GPU and CPU inference benchmark for ZyboGPT.

Measures tokens/sec for autoregressive generation on the host GPU and CPU,
comparing both float and hw_mode (ternary-quantized) inference.

Usage:
    python -m python.train.benchmark [--checkpoint PATH] [--device cuda]
"""

import argparse
import time

import torch

from .config import ZyboGPTConfig
from .model import ZyboGPT
from .tokenizer import ASCIITokenizer


@torch.no_grad()
def _decode_loop(model, tokens, gen_tokens):
    """Autoregressive decode with KV cache."""
    logits, kv = model(tokens)
    pos = tokens.shape[1]
    for i in range(gen_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        logits, kv = model(next_token, kv_caches=kv, start_pos=pos + i)


def _time_cuda(fn, repeats):
    """Time a function using CUDA events (precise GPU timing)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn(repeats)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


def _time_cpu(fn, repeats):
    """Time a function using perf_counter (wall clock)."""
    t0 = time.perf_counter()
    fn(repeats)
    return time.perf_counter() - t0


def benchmark_decode(model, device, prompt_tokens, gen_tokens, warmup=10, repeats=50):
    """Measure autoregressive decode throughput (tok/s)."""
    model.eval()
    tokens = prompt_tokens.unsqueeze(0).to(device)
    is_cuda = device.startswith("cuda")

    for _ in range(warmup):
        _decode_loop(model, tokens, gen_tokens)
    if is_cuda:
        torch.cuda.synchronize(device)

    def run(n):
        for _ in range(n):
            _decode_loop(model, tokens, gen_tokens)

    total_tokens = gen_tokens * repeats
    elapsed = _time_cuda(run, repeats) if is_cuda else _time_cpu(run, repeats)
    return total_tokens / elapsed, elapsed, total_tokens


def benchmark_prefill(model, device, batch_size, seq_len, warmup=20, repeats=200):
    """Measure prefill throughput (tok/s) — full-sequence forward pass."""
    model.eval()
    tokens = torch.randint(0, 128, (batch_size, seq_len), device=device)
    is_cuda = device.startswith("cuda")

    with torch.no_grad():
        for _ in range(warmup):
            model(tokens)
    if is_cuda:
        torch.cuda.synchronize(device)

    def run(n):
        with torch.no_grad():
            for _ in range(n):
                model(tokens)

    total_tokens = batch_size * seq_len * repeats
    elapsed = _time_cuda(run, repeats) if is_cuda else _time_cpu(run, repeats)
    return total_tokens / elapsed, elapsed, total_tokens


def run_mode(mode_name, hw_mode, device, dtype, use_compile, checkpoint, prompt_tokens, gen_tokens):
    """Benchmark a single mode (float or hw_mode). Returns results dict."""
    print(f"--- {mode_name} ---")
    config = ZyboGPTConfig(hw_mode=hw_mode)
    model = ZyboGPT(config).to(device=device, dtype=dtype)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    params = model.count_params()
    print(f"  Params: {params['total']:,} ({params['ternary']:,} ternary + {params['full_precision']:,} full)")

    if use_compile:
        model = torch.compile(model, dynamic=True)

    results = {}

    # Prefill benchmarks
    for bs in [1, 32, 256]:
        tok_s, elapsed, total = benchmark_prefill(model, device, batch_size=bs, seq_len=128)
        label = f"prefill_{bs}"
        print(f"  Prefill (B={bs:<3}, T=128): {tok_s:>12,.0f} tok/s  ({elapsed:.3f}s, {total:,} tokens)")
        results[label] = tok_s

    # Decode (autoregressive, B=1)
    tok_s, elapsed, total = benchmark_decode(model, device, prompt_tokens, gen_tokens=gen_tokens)
    print(f"  Decode  (B=1, greedy):  {tok_s:>12,.0f} tok/s  ({elapsed:.3f}s, {total:,} tokens)")
    results["decode"] = tok_s

    print()
    torch._dynamo.reset()
    return results


def run_device(device_name, device, dtype, use_compile, checkpoint, prompt_tokens, gen_tokens):
    """Run all modes for a given device. Returns dict of results."""
    print(f"=== {device_name} ===")
    print()
    float_r = run_mode("float", False, device, dtype, use_compile, checkpoint, prompt_tokens, gen_tokens)
    hw_r = run_mode("hw_mode (INT8 sim)", True, device, dtype, use_compile, checkpoint, prompt_tokens, gen_tokens)
    return float_r, hw_r


def main():
    parser = argparse.ArgumentParser(description="ZyboGPT Inference Benchmark (tok/s)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: random init)")
    parser.add_argument("--device", type=str, default="all",
                        choices=["all", "cpu", "cuda"],
                        help="Device to benchmark (default: all)")
    parser.add_argument("--gen-tokens", type=int, default=64,
                        help="Tokens to generate per decode run")
    parser.add_argument("--compile", action="store_true", default=True,
                        help="Use torch.compile on CUDA (default: True)")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    torch.set_float32_matmul_precision("high")

    print(f"ZyboGPT Inference Benchmark")
    print(f"  dtype: {args.dtype}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CPU: {torch.get_num_threads()} threads")
    print()

    tokenizer = ASCIITokenizer()
    prompt_tokens = torch.tensor(tokenizer.encode("ROMEO:"), dtype=torch.long)

    all_results = {}

    # CPU benchmark (no compile — torch.compile has high overhead on CPU for tiny models)
    if args.device in ("all", "cpu"):
        cpu_float, cpu_hw = run_device(
            "CPU", "cpu", dtype, False, args.checkpoint,
            prompt_tokens, args.gen_tokens,
        )
        all_results["cpu_float"] = cpu_float
        all_results["cpu_hw"] = cpu_hw

    # GPU benchmark
    if args.device in ("all", "cuda"):
        if not torch.cuda.is_available():
            print("CUDA not available, skipping GPU benchmark")
        else:
            gpu_float, gpu_hw = run_device(
                f"GPU ({torch.cuda.get_device_name(0)})", "cuda", dtype,
                args.compile, args.checkpoint, prompt_tokens, args.gen_tokens,
            )
            all_results["gpu_float"] = gpu_float
            all_results["gpu_hw"] = gpu_hw

    # Summary
    print("=== Summary ===")
    for key, label in [
        ("cpu_float", "CPU decode (float)"),
        ("cpu_hw", "CPU decode (hw_mode)"),
        ("gpu_float", "GPU decode (float)"),
        ("gpu_hw", "GPU decode (hw_mode)"),
    ]:
        if key in all_results:
            print(f"  {label + ':':30s} {all_results[key]['decode']:>12,.0f} tok/s")

    print()
    for key, label in [
        ("cpu_float", "CPU prefill peak (float)"),
        ("cpu_hw", "CPU prefill peak (hw_mode)"),
        ("gpu_float", "GPU prefill peak (float)"),
        ("gpu_hw", "GPU prefill peak (hw_mode)"),
    ]:
        if key in all_results:
            peak = max(all_results[key][k] for k in ("prefill_1", "prefill_32", "prefill_256"))
            print(f"  {label + ':':30s} {peak:>12,.0f} tok/s")


if __name__ == "__main__":
    main()
