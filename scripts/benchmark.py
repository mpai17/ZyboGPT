#!/usr/bin/env python3
"""ZyboGPT cross-platform benchmark: CPU vs GPU vs FPGA.

Runs autoregressive token generation on all available platforms and
prints a comparison table.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --checkpoint checkpoints/phase2/best.pt
    python scripts/benchmark.py --no-fpga          # skip FPGA
    python scripts/benchmark.py --fpga-only         # skip CPU/GPU
"""

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board import (
    find_serial_port, probe_uart_rx,
    UartBackend, XsdbBackend, SerialReader,
)

import torch
from python.train.config import ZyboGPTConfig
from python.train.model import ZyboGPT
from python.train.tokenizer import ASCIITokenizer

PROMPT = "ROMEO:"


# ── CPU / GPU benchmarks ──────────────────────────────────────


@torch.no_grad()
def bench_decode(model, device, prompt_tokens, gen_tokens, warmup=5, repeats=20):
    """Measure autoregressive decode throughput (tok/s)."""
    tokens = prompt_tokens.unsqueeze(0).to(device)

    # Warmup
    for _ in range(warmup):
        _decode_once(model, tokens, gen_tokens)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Timed runs
    if device.startswith("cuda"):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(repeats):
            _decode_once(model, tokens, gen_tokens)
        end_ev.record()
        torch.cuda.synchronize()
        elapsed = start_ev.elapsed_time(end_ev) / 1000.0
    else:
        t0 = time.perf_counter()
        for _ in range(repeats):
            _decode_once(model, tokens, gen_tokens)
        elapsed = time.perf_counter() - t0

    total_tokens = gen_tokens * repeats
    return total_tokens / elapsed


def _decode_once(model, tokens, gen_tokens):
    """Single autoregressive decode pass."""
    logits, kv = model(tokens)
    pos = tokens.shape[1]
    for i in range(gen_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        logits, kv = model(next_token, kv_caches=kv, start_pos=pos + i)


def run_host_bench(checkpoint, gen_tokens):
    """Run CPU and GPU benchmarks. Returns dict of results."""
    tokenizer = ASCIITokenizer()
    prompt_tokens = torch.tensor(tokenizer.encode(PROMPT), dtype=torch.long)

    results = {}

    # Load model
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = ckpt.get("config", ZyboGPTConfig())
        state = ckpt["model"]
    else:
        config = ZyboGPTConfig()
        state = None

    # CPU
    print("  CPU...", end="", flush=True)
    model = ZyboGPT(config).to("cpu")
    if state:
        model.load_state_dict(state)
    model.eval()
    tok_s = bench_decode(model, "cpu", prompt_tokens, gen_tokens)
    results["CPU"] = tok_s
    print(f" {tok_s:,.0f} tok/s")

    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU ({gpu_name})...", end="", flush=True)
        model = ZyboGPT(config).to("cuda")
        if state:
            model.load_state_dict(state)
        model.eval()
        model = torch.compile(model, dynamic=True)
        tok_s = bench_decode(model, "cuda", prompt_tokens, gen_tokens)
        results[f"GPU ({gpu_name})"] = tok_s
        print(f" {tok_s:,.0f} tok/s")
        torch._dynamo.reset()
    else:
        print("  GPU: not available (no CUDA)")

    return results


# ── FPGA benchmark ─────────────────────────────────────────────


def run_fpga_bench():
    """Run BENCH on the FPGA via serial. Returns tok/s or None."""
    port = find_serial_port()
    if port is None:
        print("  FPGA: no serial port found (board not connected?)")
        return None

    uart_rx_ok = probe_uart_rx(port)
    method = "serial" if uart_rx_ok else "XSDB mailbox"
    print(f"  FPGA ({port}, {method})...", end="", flush=True)

    try:
        reader = SerialReader(port)
        reader.silent = True
        reader.start()
        reader.drain(0.3)

        if uart_rx_ok:
            backend = UartBackend(reader.ser)
        else:
            backend = XsdbBackend()
            backend.connect()

        backend.send("BENCH")

        # Wait for STATS line in serial output
        deadline = time.time() + 60  # 1 min timeout
        tok_per_sec = None

        while time.time() < deadline:
            done = reader.prompt_event.is_set()
            with reader._lock:
                match = re.search(r"Tokens/sec:\s*(\d+)", reader._buf)
                if match:
                    tok_per_sec = int(match.group(1))
                    break
            if done:
                reader.drain(0.2)
                # Check once more after drain
                with reader._lock:
                    match = re.search(r"Tokens/sec:\s*(\d+)", reader._buf)
                    if match:
                        tok_per_sec = int(match.group(1))
                break
            time.sleep(0.1)

        backend.close()
        reader.stop()

        if tok_per_sec:
            print(f" {tok_per_sec:,} tok/s")
            return tok_per_sec
        else:
            print(" no STATS received (firmware not running?)")
            return None

    except Exception as e:
        print(f" error: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="ZyboGPT Cross-Platform Benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase2/best.pt",
                        help="Path to checkpoint (default: checkpoints/phase2/best.pt)")
    parser.add_argument("--gen-tokens", type=int, default=64,
                        help="Tokens to generate per decode run (default: 64)")
    parser.add_argument("--no-fpga", action="store_true",
                        help="Skip FPGA benchmark")
    parser.add_argument("--fpga-only", action="store_true",
                        help="Only run FPGA benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("  ZyboGPT Cross-Platform Benchmark")
    print("=" * 60)
    print(f"  Prompt: \"{PROMPT}\"")
    print(f"  Generate: {args.gen_tokens} tokens per run")
    print()

    results = {}

    # Host benchmarks (CPU + GPU)
    if not args.fpga_only:
        print("Host benchmarks (autoregressive decode, greedy):")
        host_results = run_host_bench(args.checkpoint, args.gen_tokens)
        results.update(host_results)
        print()

    # FPGA benchmark
    if not args.no_fpga:
        print("FPGA benchmark (128-token BENCH via serial):")
        fpga_tok_s = run_fpga_bench()
        if fpga_tok_s:
            results["FPGA (Zybo Z7-10, 150 MHz)"] = fpga_tok_s
        print()

    # Summary table
    if not results:
        print("No results collected.")
        return

    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print()

    max_tok_s = max(results.values())
    max_label_len = max(len(k) for k in results)

    for platform, tok_s in results.items():
        bar_len = int(40 * tok_s / max_tok_s) if max_tok_s > 0 else 0
        bar = "#" * bar_len
        print(f"  {platform:<{max_label_len}}  {tok_s:>10,.0f} tok/s  {bar}")

    print()

    # Speedup comparisons — all pairs
    platforms = list(results.keys())
    if len(platforms) >= 2:
        print("  Speedups:")
        for i in range(len(platforms)):
            for j in range(i + 1, len(platforms)):
                a, b = platforms[i], platforms[j]
                va, vb = results[a], results[b]
                if va > vb:
                    fast, slow, ratio = a, b, va / vb
                else:
                    fast, slow, ratio = b, a, vb / va
                print(f"    {fast} vs {slow}: {ratio:.1f}x")

    print()


if __name__ == "__main__":
    main()
