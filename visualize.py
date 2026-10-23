"""
visualize.py — Publication-quality visualisations for the VAE research project.

Figures produced:
  1. Reconstructions grid (original vs. reconstructed)
  2. Prior samples grid
  3. 2-D latent manifold traversal (Fig. 4 style from the paper)
  4. 2-D latent scatter coloured by class label
  5. Smooth interpolation between two latent codes
  6. Training curves (ELBO, KL, reconstruction loss)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reconstructions
# ─────────────────────────────────────────────────────────────────────────────

def plot_reconstructions(original, reconstructed, num_samples: int = 10,
                         save_path: str = None, title: str = "VAE"):
    """Side-by-side original (top) and reconstructed (bottom) images."""
    save_path = save_path or os.path.join(RESULTS_DIR, "reconstructions.png")
    n = min(num_samples, original.shape[0])

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    for i in range(n):
        for row, arr in enumerate([original, reconstructed]):
            img = arr[i].reshape(28, 28)
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            axes[row, i].imshow(img, cmap="gray_r", vmin=0, vmax=1)
            axes[row, i].axis("off")
        if i == 0:
            axes[0, 0].set_title("Original",      fontsize=9, pad=3)
            axes[1, 0].set_title("Reconstruction", fontsize=9, pad=3)

    fig.suptitle(f"{title} — Reconstructions", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prior samples
# ─────────────────────────────────────────────────────────────────────────────

def plot_prior_samples(model, num_samples: int = 16, device=None,
                       save_path: str = None, title: str = "VAE"):
    """Decode samples drawn from N(0, I) — tests generative quality."""
    save_path = save_path or os.path.join(RESULTS_DIR, "prior_samples.png")
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        z       = torch.randn(num_samples, model.latent_dim, device=device)
        samples = model.decoder(z).cpu().numpy().reshape(num_samples, 28, 28)

    cols = 8
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.array(axes).reshape(rows, cols)
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx < num_samples:
            ax.imshow(samples[idx], cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(f"{title} — Prior Samples p(z) ~ N(0, I)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. 2-D latent manifold (Fig. 4 of Kingma & Welling 2013)
# ─────────────────────────────────────────────────────────────────────────────

def plot_2d_latent_manifold(model, grid_size: int = 20, device=None,
                             save_path: str = None):
    """
    Sweep a 2-D grid through percentiles of N(0,1) and decode each point.
    Requires latent_dim == 2.  Skips gracefully otherwise.
    """
    save_path = save_path or os.path.join(RESULTS_DIR, "latent_manifold.png")
    if model.latent_dim != 2:
        return None   # Only meaningful for 2-D latent

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    # Map grid uniformly over [0.05, 0.95] quantiles of N(0,1)
    grid_z = norm.ppf(np.linspace(0.05, 0.95, grid_size))
    canvas = np.zeros((28 * grid_size, 28 * grid_size))

    with torch.no_grad():
        for i, yi in enumerate(grid_z):
            for j, xi in enumerate(grid_z):
                z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
                digit = model.decoder(z).cpu().numpy().reshape(28, 28)
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = digit

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap="gray_r", origin="upper")
    ax.set_xticks(np.arange(grid_size) * 28 + 14)
    ax.set_yticks(np.arange(grid_size) * 28 + 14)
    ax.set_xticklabels([f"{v:.1f}" for v in grid_z], rotation=90, fontsize=6)
    ax.set_yticklabels([f"{v:.1f}" for v in grid_z], fontsize=6)
    ax.set_title("Latent Space Manifold (Fig. 4, Kingma & Welling 2013)", fontsize=12)
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. 2-D latent scatter plot coloured by class
# ─────────────────────────────────────────────────────────────────────────────

def plot_latent_scatter(model, data_loader, device=None, max_points: int = 5000,
                        save_path: str = None, title: str = "VAE"):
    """Encode test images → scatter plot coloured by digit class."""
    save_path = save_path or os.path.join(RESULTS_DIR, "latent_scatter.png")
    if model.latent_dim != 2:
        return None

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    mus, labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            imgs, lbls = batch[0].to(device), batch[1]
            mu, _ = model.encoder(imgs)
            mus.append(mu.cpu())
            labels.append(lbls)
            if sum(len(l) for l in labels) >= max_points:
                break

    mus    = torch.cat(mus   )[:max_points].numpy()
    labels = torch.cat(labels)[:max_points].numpy()

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in range(10):
        mask = labels == cls
        ax.scatter(mus[mask, 0], mus[mask, 1], s=4, alpha=0.6,
                   color=cmap(cls / 10), label=str(cls))
    ax.legend(markerscale=4, title="Digit", fontsize=8)
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.set_title(f"{title} — Latent Space (2D)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Latent interpolation
# ─────────────────────────────────────────────────────────────────────────────

def plot_interpolation(model, x_a: torch.Tensor, x_b: torch.Tensor,
                       steps: int = 10, device=None,
                       save_path: str = None, title: str = "VAE"):
    """Linear interpolation in latent space between two encoded images."""
    save_path = save_path or os.path.join(RESULTS_DIR, "interpolation.png")
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        mu_a, _ = model.encoder(x_a.unsqueeze(0).to(device))
        mu_b, _ = model.encoder(x_b.unsqueeze(0).to(device))

        alphas  = np.linspace(0, 1, steps)
        decoded = []
        for a in alphas:
            z       = (1 - a) * mu_a + a * mu_b
            decoded.append(model.decoder(z).cpu().numpy().reshape(28, 28))

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.5, 2))
    for i, ax in enumerate(axes):
        ax.imshow(decoded[i], cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title("Source", fontsize=8)
        if i == steps - 1:
            ax.set_title("Target", fontsize=8)
    fig.suptitle(f"{title} — Latent Interpolation", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: list[dict], save_path: str = None, title: str = "VAE"):
    """
    Plot ELBO, KL divergence, and KL annealing factor over training epochs.
    `history` is the list of dicts returned by train_and_evaluate.
    """
    save_path = save_path or os.path.join(RESULTS_DIR, "training_curves.png")
    epochs     = [h["epoch"]      for h in history]
    train_elbo = [h["train_elbo"] for h in history]
    val_elbo   = [h["val_elbo"]   for h in history]
    kl         = [h["kl"]         for h in history]
    kl_factor  = [h.get("kl_factor", 1.0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_elbo, label="Train")
    axes[0].plot(epochs, val_elbo,   label="Val",  linestyle="--")
    axes[0].set_title("Negative ELBO")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, kl, color="tab:orange")
    axes[1].set_title("KL Divergence")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("KL (nats/sample)")

    axes[2].plot(epochs, kl_factor, color="tab:green")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("KL Annealing Factor")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("β factor")

    fig.suptitle(f"{title} — Training Curves", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. Quantitative model comparison bar-chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(summary_df, save_path: str = None):
    """Bar-chart comparing ELBO, NLL, KL across models (mean ± std)."""
    save_path = save_path or os.path.join(RESULTS_DIR, "model_comparison.png")

    metrics = {
        "ELBO ↓":   ("elbo_mean",       "elbo_std"),
        "NLL ↓":    ("recon_nll_mean",   "recon_nll_std"),
        "KL ↑":     ("kl_div_mean",      "kl_div_std"),
        "IW-LL ↑":  ("iw_log_likelihood_mean", "iw_log_likelihood_std"),
    }

    available = {k: v for k, v in metrics.items()
                 if v[0] in summary_df.columns}
    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 4, 4.5))
    if n_metrics == 1:
        axes = [axes]

    models = summary_df.index.tolist()
    colors = ["#4C9BE8", "#E86C4C", "#5ABE7A"]
    x      = np.arange(len(models))

    for ax, (label, (mean_col, std_col)) in zip(axes, available.items()):
        means = summary_df[mean_col].values
        stds  = summary_df[std_col].values if std_col in summary_df.columns else np.zeros_like(means)
        bars  = ax.bar(x, means, yerr=stds, color=colors[:len(models)],
                       capsize=5, width=0.55, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[0] * 0.1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Comparison — Section 5 Experiments", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path