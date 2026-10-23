"""
tests/test_vae.py — Unit tests for models and metrics.

Covers:
  • Forward-pass output shapes
  • Reparameterisation gradient flow
  • KL non-negativity (Gibbs inequality)
  • Importance-weighted log-likelihood execution
  • VanillaAE forward pass
  • Checkpoint save / load round-trip
"""

import os
import tempfile
import math
import pytest
import torch
from models import VariationalAutoencoder, VanillaAutoencoder
from metrics import vae_loss_function, importance_weighted_log_likelihood


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_vae():
    return VariationalAutoencoder(input_dim=784, hidden_dims=[128, 64], latent_dim=10)


@pytest.fixture
def small_ae():
    return VanillaAutoencoder(input_dim=784, hidden_dims=[128, 64], latent_dim=10)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(16, 784)


# ─────────────────────────────────────────────────────────────────────────────
# Model shape tests
# ─────────────────────────────────────────────────────────────────────────────

def test_vae_forward_shapes(small_vae, batch):
    x_recon, mu, logvar = small_vae(batch)
    assert x_recon.shape == (16, 784),  f"x_recon shape mismatch: {x_recon.shape}"
    assert mu.shape     == (16,  10),   f"mu shape mismatch: {mu.shape}"
    assert logvar.shape == (16,  10),   f"logvar shape mismatch: {logvar.shape}"


def test_vae_output_range(small_vae, batch):
    """Sigmoid decoder must output values in (0, 1)."""
    x_recon, _, _ = small_vae(batch)
    assert x_recon.min().item() >= 0.0 - 1e-6
    assert x_recon.max().item() <= 1.0 + 1e-6


def test_ae_forward_shapes(small_ae, batch):
    x_recon, z = small_ae(batch)
    assert x_recon.shape == (16, 784), f"AE x_recon shape mismatch: {x_recon.shape}"
    assert z.shape       == (16,  10), f"AE z shape mismatch: {z.shape}"


# ─────────────────────────────────────────────────────────────────────────────
# Reparameterisation
# ─────────────────────────────────────────────────────────────────────────────

def test_reparameterisation_gradient_flow(small_vae):
    """Gradients must flow back through the reparameterised sample to encoder weights."""
    x = torch.rand(8, 784, requires_grad=False)
    x_recon, mu, logvar = small_vae(x)
    loss, _, _ = vae_loss_function(x_recon, x, mu, logvar)
    loss.backward()

    assert small_vae.encoder.fc_mu.weight.grad is not None, "No grad on fc_mu.weight"
    assert not torch.isnan(small_vae.encoder.fc_mu.weight.grad).any(), "NaN grad in fc_mu"


def test_reparameterisation_eval_is_deterministic(small_vae, batch):
    """In eval mode, two forward passes from the same input must be identical."""
    small_vae.eval()
    with torch.no_grad():
        _, mu1, _ = small_vae(batch)
        _, mu2, _ = small_vae(batch)
    assert torch.allclose(mu1, mu2)


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

def test_kl_divergence_non_negativity():
    """KL[N(μ,σ²) || N(0,1)] ≥ 0 (Gibbs inequality)."""
    torch.manual_seed(0)
    mu, logvar = torch.randn(32, 10), torch.randn(32, 10)
    x, x_recon = torch.rand(32, 784), torch.rand(32, 784)
    _, _, kl = vae_loss_function(x_recon, x, mu, logvar)
    assert kl.item() >= -1e-4, f"KL should be ≥ 0, got {kl.item()}"


def test_elbo_equals_nll_plus_kl():
    """Total ELBO must exactly equal NLL + β·KL."""
    torch.manual_seed(1)
    mu, logvar = torch.randn(8, 5), torch.randn(8, 5)
    x, x_recon = torch.rand(8, 784), torch.rand(8, 784)
    for beta in [1.0, 4.0]:
        elbo, nll, kl = vae_loss_function(x_recon, x, mu, logvar, beta=beta)
        expected = nll + beta * kl
        assert torch.isclose(elbo, expected, atol=1e-4), (
            f"ELBO ≠ NLL + β·KL for β={beta}: {elbo.item():.4f} vs {expected.item():.4f}"
        )


def test_kl_zero_for_standard_normal():
    """KL[N(0, I) || N(0, I)] should be ≈ 0."""
    mu     = torch.zeros(16, 10)
    logvar = torch.zeros(16, 10)  # log(1) = 0
    x, x_recon = torch.rand(16, 784), torch.rand(16, 784)
    _, _, kl = vae_loss_function(x_recon, x, mu, logvar)
    assert abs(kl.item()) < 1e-4, f"KL should be ~0 for standard normal, got {kl.item()}"


# ─────────────────────────────────────────────────────────────────────────────
# Importance-Weighted Log-Likelihood
# ─────────────────────────────────────────────────────────────────────────────

def test_importance_weighted_ll_returns_float(small_vae):
    x = torch.rand(4, 784)
    ll = importance_weighted_log_likelihood(small_vae, x, num_samples=20)
    assert isinstance(ll, float), f"IW-LL should be float, got {type(ll)}"
    assert math.isfinite(ll),     f"IW-LL should be finite, got {ll}"


def test_importance_weighted_ll_is_negative(small_vae):
    """log p(x) is expected to be negative for this model class."""
    x = torch.rand(8, 784)
    ll = importance_weighted_log_likelihood(small_vae, x, num_samples=50)
    assert ll < 0, f"IW-LL should be negative (log-prob), got {ll}"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_save_load(small_vae, batch, tmp_path):
    """Saving and loading a checkpoint must preserve model outputs."""
    small_vae.eval()
    ckpt_path = str(tmp_path / "test_vae.pt")

    with torch.no_grad():
        out_before, _, _ = small_vae(batch)

    torch.save({"model_state": small_vae.state_dict()}, ckpt_path)

    # New model with same arch
    new_model = VariationalAutoencoder(input_dim=784, hidden_dims=[128, 64], latent_dim=10)
    ckpt = torch.load(ckpt_path, weights_only=True)
    new_model.load_state_dict(ckpt["model_state"])
    new_model.eval()

    with torch.no_grad():
        out_after, _, _ = new_model(batch)

    assert torch.allclose(out_before, out_after, atol=1e-6), "Outputs differ after checkpoint round-trip"


# ─────────────────────────────────────────────────────────────────────────────
# Prior sampling
# ─────────────────────────────────────────────────────────────────────────────

def test_prior_sampling_shape(small_vae):
    samples = small_vae.sample(32, device=torch.device("cpu"))
    assert samples.shape == (32, 784), f"Prior sample shape mismatch: {samples.shape}"
    assert samples.min().item() >= 0.0 - 1e-6
    assert samples.max().item() <= 1.0 + 1e-6