"""
train.py — Full training + evaluation pipeline for the VAE research project.

Key improvements vs. baseline:
  • Real binarised MNIST via torchvision (falls back to Fashion-MNIST)
  • KL annealing (warm-up) to prevent posterior collapse
  • Cosine-annealing LR schedule
  • Gradient clipping
  • Early stopping on validation ELBO
  • Checkpoint saving / loading
  • Multi-seed experiments with saved CSV summary
"""

import os
import json
import time
import warnings
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from torchvision import datasets, transforms

from models import VariationalAutoencoder, VanillaAutoencoder
from metrics import vae_loss_function, importance_weighted_log_likelihood
from utils import set_seed, count_parameters, profile_latency_and_memory

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "experiment_summary.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)


def get_binarised_mnist(batch_size: int = 128, val_fraction: float = 0.1):
    """
    Returns (train_loader, val_loader, test_loader) for binarised MNIST.
    Images are thresholded at 0.5 to produce binary pixels,
    matching the Bernoulli observation model in the paper.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float().view(-1)),  # binarise + flatten
    ])

    try:
        train_full = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
        test_set   = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
        dataset_name = "MNIST"
    except Exception:
        warnings.warn("MNIST download failed — falling back to FashionMNIST.")
        train_full = datasets.FashionMNIST(DATA_DIR, train=True,  download=True, transform=transform)
        test_set   = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
        dataset_name = "FashionMNIST"

    n_val = int(len(train_full) * val_fraction)
    n_train = len(train_full) - n_val
    train_set, val_set = random_split(train_full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=0)

    print(f"Dataset: {dataset_name}  | train={n_train} val={n_val} test={len(test_set)}")
    return train_loader, val_loader, test_loader, dataset_name


# ─────────────────────────────────────────────────────────────────────────────
# Single training run
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    model_type: str = "vae",
    latent_dim: int = 20,
    beta: float = 1.0,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    kl_warmup_epochs: int = 10,
    patience: int = 7,
    hidden_dims: list[int] = None,
    save_checkpoint: bool = True,
):
    """
    Train a model and return a dict of evaluation metrics.

    KL annealing: β is linearly ramped from 0 → target over `kl_warmup_epochs`.
    This prevents early posterior collapse (KL vanishing to 0).
    """
    set_seed(seed)
    if hidden_dims is None:
        hidden_dims = [512, 256]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Model: {model_type} | latent_dim={latent_dim} | beta={beta} | seed={seed} | device={device}")

    train_loader, val_loader, test_loader, _ = get_binarised_mnist(batch_size)

    # ── Build model ────────────────────────────────────────────────────────
    if model_type in ("vae", "beta_vae"):
        model = VariationalAutoencoder(
            input_dim=784, hidden_dims=hidden_dims,
            latent_dim=latent_dim, beta=beta,
        ).to(device)
    else:
        model = VanillaAutoencoder(
            input_dim=784, hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Mixed precision (CPU-safe)
    use_amp   = device.type == "cuda"
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_elbo = float("inf")
    patience_counter = 0
    train_history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # KL annealing factor (0 → 1 linearly over warm-up epochs)
        kl_factor = min(1.0, epoch / max(1, kl_warmup_epochs))

        model.train()
        train_elbo_sum = train_nll_sum = train_kl_sum = 0.0

        for batch in train_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                if model_type == "vanilla_ae":
                    x_recon, _ = model(x)
                    loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
                    nll, kl = loss.item(), 0.0
                else:
                    x_recon, mu, logvar = model(x)
                    # Use annealed beta for this epoch
                    annealed_beta = beta * kl_factor
                    loss, nll, kl = vae_loss_function(x_recon, x, mu, logvar, beta=annealed_beta)

            scaler.scale(loss).backward()
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_elbo_sum += loss.item()
            train_nll_sum  += (nll if isinstance(nll, float) else nll.item())
            train_kl_sum   += (kl  if isinstance(kl,  float) else kl.item())

        scheduler.step()

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_elbo_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                if model_type == "vanilla_ae":
                    x_recon, _ = model(x)
                    val_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
                else:
                    x_recon, mu, logvar = model(x)
                    val_loss, _, _ = vae_loss_function(x_recon, x, mu, logvar, beta=beta)
                val_elbo_sum += val_loss.item()

        avg_val_elbo   = val_elbo_sum   / len(val_loader)
        avg_train_elbo = train_elbo_sum / len(train_loader)
        avg_kl         = train_kl_sum   / len(train_loader)

        train_history.append({
            "epoch": epoch,
            "train_elbo": avg_train_elbo,
            "val_elbo":   avg_val_elbo,
            "kl":         avg_kl,
            "kl_factor":  kl_factor,
        })

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"train_ELBO={avg_train_elbo:.2f} | val_ELBO={avg_val_elbo:.2f} | "
              f"KL={avg_kl:.2f} | kl_factor={kl_factor:.2f}")

        # ── Early stopping + checkpointing ───────────────────────────────────
        if avg_val_elbo < best_val_elbo - 1e-3:
            best_val_elbo    = avg_val_elbo
            patience_counter = 0
            if save_checkpoint:
                ckpt_path = os.path.join(
                    CKPT_DIR,
                    f"{model_type}_ld{latent_dim}_seed{seed}.pt"
                )
                torch.save({
                    "epoch":      epoch,
                    "model_state": model.state_dict(),
                    "val_elbo":   best_val_elbo,
                    "config": {
                        "model_type":  model_type,
                        "latent_dim":  latent_dim,
                        "hidden_dims": hidden_dims,
                        "beta":        beta,
                    },
                }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    total_train_time = time.time() - start_time

    # ── Load best checkpoint for evaluation ─────────────────────────────────
    if save_checkpoint:
        ckpt_path = os.path.join(CKPT_DIR, f"{model_type}_ld{latent_dim}_seed{seed}.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded best checkpoint (val_ELBO={ckpt['val_elbo']:.2f}) "
                  f"from epoch {ckpt['epoch']}")

    # ── Test-set evaluation ──────────────────────────────────────────────────
    model.eval()
    test_elbo_sum = test_nll_sum = test_kl_sum = 0.0
    all_test_x = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            all_test_x.append(x)
            if model_type == "vanilla_ae":
                x_recon, _ = model(x)
                nll_b = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
                test_elbo_sum += nll_b.item()
                test_nll_sum  += nll_b.item()
            else:
                x_recon, mu, logvar = model(x)
                elbo_b, nll_b, kl_b = vae_loss_function(x_recon, x, mu, logvar, beta=beta)
                test_elbo_sum += elbo_b.item()
                test_nll_sum  += nll_b.item()
                test_kl_sum   += kl_b.item()

    n_batches = len(test_loader)
    test_elbo = test_elbo_sum / n_batches
    test_nll  = test_nll_sum  / n_batches
    test_kl   = test_kl_sum   / n_batches

    # IW log-likelihood on a 512-sample subset
    test_x_sample = torch.cat(all_test_x)[:512]
    if model_type != "vanilla_ae":
        iw_ll = importance_weighted_log_likelihood(model, test_x_sample, num_samples=200)
    else:
        iw_ll = -test_nll

    # MSE / MAE on first batch
    with torch.no_grad():
        x_b = all_test_x[0]
        if model_type == "vanilla_ae":
            x_recon_b, _ = model(x_b)
        else:
            x_recon_b, _, _ = model(x_b)
        mse = F.mse_loss(x_recon_b, x_b).item()
        mae = F.l1_loss (x_recon_b, x_b).item()

    latency_ms, mem_mb = profile_latency_and_memory(model, x_b[:1])

    print(f"  ✓ Test ELBO={test_elbo:.2f} | NLL={test_nll:.2f} | "
          f"KL={test_kl:.2f} | IW-LL={iw_ll:.2f}")

    result = {
        "model":             model_type,
        "seed":              seed,
        "latent_dim":        latent_dim,
        "beta":              beta,
        "elbo":              test_elbo,
        "recon_nll":         test_nll,
        "kl_div":            test_kl,
        "iw_log_likelihood": iw_ll,
        "mse":               mse,
        "mae":               mae,
        "params":            count_parameters(model),
        "train_time_sec":    total_train_time,
        "latency_ms":        latency_ms,
        "mem_mb":            mem_mb,
        "best_val_elbo":     best_val_elbo,
    }

    # Save per-run history alongside checkpoint
    hist_path = os.path.join(CKPT_DIR, f"{model_type}_ld{latent_dim}_seed{seed}_history.json")
    with open(hist_path, "w") as f:
        json.dump(train_history, f, indent=2)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed experiment runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  VAE Research Project — Multi-Seed Experiments")
    print("  Reference: Kingma & Welling (2013) Section 5")
    print("=" * 70)

    experiments = [
        # (model_type, latent_dim, beta)
        ("vae",       20, 1.0),
        ("beta_vae",  20, 4.0),
        ("vanilla_ae", 20, 0.0),
    ]
    seeds  = [42, 43, 44]
    results = []

    for model_name, ld, bv in experiments:
        for seed in seeds:
            res = train_and_evaluate(
                model_type=model_name,
                latent_dim=ld,
                beta=bv,
                seed=seed,
                epochs=50,
                batch_size=128,
                lr=1e-3,
                kl_warmup_epochs=10,
                patience=7,
                save_checkpoint=True,
            )
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)

    summary_cols = ["elbo", "recon_nll", "kl_div", "iw_log_likelihood", "mse", "latency_ms"]
    summary = df.groupby("model")[summary_cols].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]

    print("\n" + "=" * 70)
    print("  MULTI-SEED EXPERIMENTAL RESULTS (mean ± std across 3 seeds)")
    print("=" * 70)
    print(summary.to_string())
    print(f"\nFull results saved → {RESULTS_PATH}")