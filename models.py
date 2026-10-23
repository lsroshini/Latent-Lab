"""
models.py — VAE, Beta-VAE, and Vanilla Autoencoder architectures.

Implements the probabilistic encoder/decoder from:
  Kingma & Welling (2013) "Auto-Encoding Variational Bayes"
  https://arxiv.org/abs/1312.6114

Architecture improvements over naive MLP:
  • Deeper networks (784 → 512 → 256 → latent_dim)
  • Batch normalisation for stable training
  • Leaky ReLU to avoid dying neurons
  • Weight initialisation via Xavier uniform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(layer_dims: list[int], activation=None, bn: bool = True) -> nn.Sequential:
    """Helper: build a sequential MLP with optional BatchNorm and activation.
    BN and activation are applied after every hidden layer but NOT the final output layer.
    """
    layers = []
    n_hidden = len(layer_dims) - 2          # number of hidden transitions
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        is_hidden = i < n_hidden             # True for all but the last linear
        if bn and is_hidden:
            layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
        if activation is not None and is_hidden:
            layers.append(activation())
    return nn.Sequential(*layers)


class ProbabilisticEncoder(nn.Module):
    """
    q_φ(z|x): Gaussian inference network.
    Outputs (μ, log σ²) for the variational posterior.
    """
    def __init__(self, input_dim: int = 784, hidden_dims: list[int] = None,
                 latent_dim: int = 20):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.net = _build_mlp(
            [input_dim] + hidden_dims,
            activation=lambda: nn.LeakyReLU(0.2),
            bn=True,
        )
        self.fc_mu     = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        h = F.leaky_relu(self.net(x), 0.2)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, -10.0, 4.0)
        return mu, logvar


class ProbabilisticDecoder(nn.Module):
    """
    p_θ(x|z): Bernoulli generative network.
    Outputs pixel probabilities via sigmoid.
    """
    def __init__(self, latent_dim: int = 20, hidden_dims: list[int] = None,
                 output_dim: int = 784):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512]

        dims = [latent_dim] + hidden_dims
        self.net = _build_mlp(dims, activation=lambda: nn.LeakyReLU(0.2), bn=True)
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z can be (B, latent_dim) or (B, S, latent_dim) for IW estimation
        orig_shape = z.shape
        if z.dim() == 3:
            B, S, D = z.shape
            z = z.reshape(B * S, D)

        h = F.leaky_relu(self.net(z), 0.2)
        x_recon = torch.sigmoid(self.fc_out(h))

        if len(orig_shape) == 3:
            x_recon = x_recon.reshape(B, S, -1)
        return x_recon


class VariationalAutoencoder(nn.Module):
    """
    Full VAE / β-VAE model (β=1 reduces to standard VAE).

    Architecture matches Section 5.3 of Kingma & Welling (2013):
      • MLP encoder/decoder
      • Gaussian latent space with diagonal covariance
      • Reparameterisation trick for back-propagation
    """
    def __init__(self, input_dim: int = 784,
                 hidden_dims: list[int] = None,
                 latent_dim: int = 20,
                 beta: float = 1.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.latent_dim = latent_dim
        self.beta       = beta
        self.encoder    = ProbabilisticEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder    = ProbabilisticDecoder(latent_dim, hidden_dims[::-1], input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick (Kingma & Welling eq. 9):
          z = μ + σ ⊙ ε,  ε ~ N(0, I)
        Uses training flag to return μ deterministically at test time.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor):
        mu, logvar  = self.encoder(x)
        z           = self.reparameterize(mu, logvar)
        x_recon     = self.decoder(z)
        return x_recon, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Draw samples from the prior p(z) = N(0, I) and decode."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decoder(z)


class VanillaAutoencoder(nn.Module):
    """
    Deterministic AE baseline for Section 5 comparisons.
    Same capacity as VAE for fair comparison.
    """
    def __init__(self, input_dim: int = 784,
                 hidden_dims: list[int] = None,
                 latent_dim: int = 20):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.latent_dim = latent_dim
        enc_dims = [input_dim] + hidden_dims + [latent_dim]
        dec_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]

        self.encoder = _build_mlp(enc_dims, activation=lambda: nn.LeakyReLU(0.2), bn=True)
        self.decoder = _build_mlp(dec_dims, activation=lambda: nn.LeakyReLU(0.2), bn=True)

    def forward(self, x: torch.Tensor):
        z     = self.encoder(x)
        x_out = torch.sigmoid(self.decoder(z))   # decode z, not x
        return x_out, z