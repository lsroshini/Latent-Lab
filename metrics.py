"""
metrics.py — Mathematically correct ELBO loss and IW log-likelihood estimator.

Reference: Kingma & Welling (2013), Appendix B (KL) and Section 5 (IW-LL).
"""

import math
import torch
import torch.nn.functional as F


def vae_loss_function(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    likelihood: str = "bernoulli",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Negative ELBO = E_q[-log p(x|z)] + β · KL[q(z|x) || p(z)]

    The reconstruction term is summed over pixels and averaged over the batch
    (per-sample negative log-likelihood), matching the paper's per-datapoint ELBO.

    Args:
        x_recon:    Reconstructed image probabilities, shape (B, D).
        x:          Ground-truth images, shape (B, D).
        mu:         Posterior mean,    shape (B, L).
        logvar:     Posterior log-var, shape (B, L).
        beta:       KL weight (1.0 for standard VAE, >1 for β-VAE).
        likelihood: "bernoulli" (binary cross-entropy) or "gaussian" (MSE).

    Returns:
        (elbo_loss, recon_loss, kl_loss) — all per-sample scalars.
    """
    batch_size = x.size(0)

    if likelihood == "bernoulli":
        # Numerically stable binary cross-entropy, summed over pixels
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / batch_size
    elif likelihood == "gaussian":
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / batch_size
    else:
        raise ValueError(f"Unknown likelihood: {likelihood!r}")

    # Analytical KL: -½ Σ(1 + log σ² - μ² - σ²)  — Appendix B
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    elbo_loss = recon_loss + beta * kl_loss
    return elbo_loss, recon_loss, kl_loss


def importance_weighted_log_likelihood(
    model: torch.nn.Module,
    x: torch.Tensor,
    num_samples: int = 200,
) -> float:
    """
    Importance-Weighted log p(x) estimator (Burda et al. 2015, eq. 8).

    log p(x) ≈ log (1/S Σ_s  p(x|z_s) p(z_s) / q(z_s|x) ),
    where z_s ~ q(z|x).

    This is an unbiased estimator of log p(x) and tighter than the ELBO.
    """
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encoder(x)           # (B, L)
        std        = torch.exp(0.5 * logvar)
        B, L       = mu.shape
        D          = x.shape[1]

        # Sample S latent vectors per data-point: (B, S, L)
        eps = torch.randn(B, num_samples, L, device=x.device)
        z   = mu.unsqueeze(1) + eps * std.unsqueeze(1)

        # Decode all B·S vectors at once
        x_exp   = x.unsqueeze(1).expand(B, num_samples, D)  # (B, S, D)
        x_recon = model.decoder(z)                           # (B, S, D)

        eps_c = 1e-8

        # --- log p(x | z) — Bernoulli ---
        log_p_x_z = (
            x_exp * torch.log(x_recon + eps_c)
            + (1 - x_exp) * torch.log(1 - x_recon + eps_c)
        ).sum(-1)  # (B, S)

        # --- log p(z) — standard Normal N(0, I) ---
        log_p_z = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)  # (B, S)

        # --- log q(z | x) — diagonal Gaussian ---
        log_q_z = -0.5 * (
            logvar.unsqueeze(1)
            + (z - mu.unsqueeze(1)).pow(2) / (std.unsqueeze(1).pow(2) + eps_c)
            + math.log(2 * math.pi)
        ).sum(-1)  # (B, S)

        # Importance weights (log scale)
        log_w = log_p_x_z + log_p_z - log_q_z  # (B, S)

        # Log-mean-exp for numerical stability
        log_p_x = torch.logsumexp(log_w, dim=1) - math.log(num_samples)  # (B,)

    return log_p_x.mean().item()