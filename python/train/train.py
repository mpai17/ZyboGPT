"""Training loop for ZyboGPT.

Usage:
    python -m python.train.train [--steps 50000] [--device cuda]
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ZyboGPTConfig
from .dataset import ShakespeareDataset
from .bitlinear import set_hw_truncation
from .model import ZyboGPT
from .tokenizer import ASCIITokenizer


def get_lr(step: int, max_steps: int, warmup_steps: int, lr: float) -> float:
    """Linear warmup + cosine decay schedule."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr * max(coeff, 0.01)


def train(config=None, device="auto", save_dir="checkpoints", steps=None,
          label_smoothing=0.0, truncation_step=None, resume=None,
          continue_from=None):
    torch.set_float32_matmul_precision("high")
    if config is None:
        config = ZyboGPTConfig()
    if steps is not None:
        config.max_steps = steps

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mode_str = "hw_mode" if config.hw_mode else "float"
    print(f"Training ZyboGPT on {device} ({mode_str})")
    print(f"  vocab={config.vocab_size}, d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"  n_heads={config.n_heads}, d_ff={config.d_ff}, ctx_len={config.ctx_len}")
    if config.hw_mode:
        print(f"  hw_mode=True, label_smoothing={label_smoothing}")
        if truncation_step is not None:
            print(f"  curriculum: clamp Phase 1 (0-{truncation_step}), truncate Phase 2 ({truncation_step}-{config.max_steps})")
            set_hw_truncation(False)  # Start with clamping
        else:
            set_hw_truncation(False)  # Default: clamping only

    model = ZyboGPT(config).to(device)

    if resume:
        print(f"  Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    params = model.count_params()
    print(f"  Parameters: {params['ternary']} ternary + {params['full_precision']} full = {params['total']}")

    dataset = ShakespeareDataset(ctx_len=config.ctx_len, split="train")
    val_dataset = ShakespeareDataset(ctx_len=config.ctx_len, split="val")
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    tokenizer = ASCIITokenizer()
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.95),
    )

    step = 0
    best_val_loss = float("inf")

    if continue_from:
        print(f"  Continuing from checkpoint: {continue_from}")
        ckpt = torch.load(continue_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resuming from step {step}, best_val_loss={best_val_loss:.4f}")

    # Compile after all checkpoint loading (compile wraps keys with _orig_mod. prefix)
    if device == "cuda":
        model = torch.compile(model)
        print(f"  torch.compile enabled")
    # Get the unwrapped model for saving (torch.compile wraps in OptimizedModule)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    data_iter = iter(loader)
    start_time = time.time()
    log_interval = min(500, config.max_steps // 10)
    val_interval = min(5000, config.max_steps // 5)

    model.train()
    while step < config.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Curriculum: switch from clamp to truncate at truncation_step
        if truncation_step is not None and step == truncation_step:
            print(f"\n  >>> Switching to truncation mode at step {step} <<<\n")
            set_hw_truncation(True)

        cur_lr = get_lr(step, config.max_steps, config.warmup_steps, config.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1),
                               label_smoothing=label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if step % max(log_interval, 1) == 0:
            elapsed = time.time() - start_time
            print(f"step {step:6d}/{config.max_steps} | loss {loss.item():.4f} | lr {cur_lr:.2e} | {elapsed:.1f}s")

        if step % max(val_interval, 1) == 0 and step > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vlogits, _ = model(vx)
                    vloss = F.cross_entropy(vlogits.view(-1, config.vocab_size), vy.view(-1))
                    val_losses.append(vloss.item())
                    if len(val_losses) >= 20:
                        break
            avg_val = sum(val_losses) / len(val_losses)
            print(f"  val_loss: {avg_val:.4f}")

            prompt = "ROMEO:"
            tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
            generated = raw_model.generate(tokens, max_new_tokens=40, temperature=0.8)
            print(f"  sample: {tokenizer.decode(generated)[:120]}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(
                    {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": config, "step": step, "best_val_loss": best_val_loss},
                    os.path.join(save_dir, "best.pt"),
                )
                print(f"  saved best model (val_loss={best_val_loss:.4f})")

            model.train()

        step += 1

    # Save final
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "config": config, "step": config.max_steps, "best_val_loss": best_val_loss},
        os.path.join(save_dir, "final.pt"),
    )
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None,
                        help="Training steps (default: 50000 normal, 150000 hw-mode)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--hw-mode", action="store_true",
                        help="Train with hardware-accurate integer arithmetic")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 5e-4 normal, 3e-4 hw-mode)")
    parser.add_argument("--truncation-step", type=int, default=None,
                        help="Step to switch from clamp to truncate (curriculum training)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (for fine-tuning, fresh optimizer)")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Continue training from checkpoint (restores optimizer + step)")
    args = parser.parse_args()

    hw_mode = args.hw_mode
    steps = args.steps or (150_000 if hw_mode else 50_000)
    lr = args.lr or (3e-4 if hw_mode else 5e-4)
    warmup = 2000 if hw_mode else 1000
    label_smoothing = 0.1 if hw_mode else 0.0

    config = ZyboGPTConfig(
        max_steps=steps,
        learning_rate=lr,
        warmup_steps=warmup,
        hw_mode=hw_mode,
    )
    train(config, device=args.device, save_dir=args.save_dir,
          label_smoothing=label_smoothing,
          truncation_step=args.truncation_step,
          resume=args.resume,
          continue_from=args.continue_from)
