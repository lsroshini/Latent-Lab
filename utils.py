"""
utils.py — Reproducibility helpers, model profiling, and checkpoint I/O.
"""

import os
import json
import random
import time
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Model introspection
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: torch.nn.Module) -> str:
    """Return a human-readable parameter count string."""
    total = count_parameters(model)
    return f"{model.__class__.__name__}: {total:,} trainable parameters"


# ─────────────────────────────────────────────────────────────────────────────
# Latency & memory profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_latency_and_memory(model: torch.nn.Module, sample_input: torch.Tensor,
                                num_runs: int = 100) -> tuple[float, float]:
    """
    Measure mean per-sample inference latency (ms) and GPU memory delta (MB).
    On CPU, memory delta is always 0.
    """
    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    # Warm-up to trigger JIT / cuDNN auto-tuning
    with torch.no_grad():
        for _ in range(10):
            model(sample_input)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            model(sample_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    latency_ms = ((end - start) / num_runs) * 1000.0

    if device.type == "cuda":
        peak_mem_mb = (torch.cuda.memory_allocated() - start_mem) / (1024 ** 2)
    else:
        peak_mem_mb = 0.0

    return latency_ms, peak_mem_mb


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(model: torch.nn.Module, ckpt_path: str,
                    device: torch.device = None) -> dict:
    """
    Load a checkpoint saved by train.py into `model`.
    Returns the metadata dict stored in the checkpoint.
    """
    if device is None:
        device = next(model.parameters()).device
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return ckpt


def find_best_checkpoint(ckpt_dir: str, model_type: str,
                          latent_dim: int, seed: int = 42) -> str | None:
    """Return path to best checkpoint for the given configuration, or None."""
    fname = os.path.join(ckpt_dir, f"{model_type}_ld{latent_dim}_seed{seed}.pt")
    return fname if os.path.exists(fname) else None


def load_training_history(ckpt_dir: str, model_type: str,
                           latent_dim: int, seed: int = 42) -> list[dict]:
    """Load the per-epoch training history JSON for a run."""
    fname = os.path.join(ckpt_dir,
                         f"{model_type}_ld{latent_dim}_seed{seed}_history.json")
    if not os.path.exists(fname):
        return []
    with open(fname) as f:
        return json.load(f)