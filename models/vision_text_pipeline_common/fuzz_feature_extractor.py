import torch
import torch.nn as nn


class FuzzyFeatureExtractor(nn.Module):
    """Context-Gated Calibration (CGC) gating score computation.

    Computes G(h) = w_G·μ_G(h) + w_S·μ_S(h) + w_T·μ_T(h) + b  (Eq. 7)
    All membership function parameters and combination weights are learnable.
    """

    def __init__(self, mu_params, sigma_params, trapezoidal_params, weights):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(float(mu_params['mu'])))
        self.sigma = nn.Parameter(torch.tensor(float(mu_params['sigma'])))
        self.alpha = nn.Parameter(torch.tensor(float(sigma_params['alpha'])))
        self.beta = nn.Parameter(torch.tensor(float(sigma_params['beta'])))
        self.a = nn.Parameter(torch.tensor(float(trapezoidal_params['a'])))
        self.b = nn.Parameter(torch.tensor(float(trapezoidal_params['b'])))
        self.c = nn.Parameter(torch.tensor(float(trapezoidal_params['c'])))
        self.d = nn.Parameter(torch.tensor(float(trapezoidal_params['d'])))
        self.w_mu = nn.Parameter(torch.tensor(float(weights['w_mu'])))
        self.w_sigma = nn.Parameter(torch.tensor(float(weights['w_sigma'])))
        self.w_T = nn.Parameter(torch.tensor(float(weights['w_T'])))
        self.b_weight = nn.Parameter(torch.tensor(float(weights['b'])))

    def gaussian(self, x):
        # Eq. 4: μ_G(h) = exp(-(h-μ)² / (2σ²))
        sigma = self.sigma.abs().clamp(min=1e-8)
        return torch.exp(-0.5 * ((x - self.mu) / sigma) ** 2)

    def sigmoid_mf(self, x):
        # Eq. 5: μ_S(h) = 1 / (1 + exp(-α(h-β)))
        return torch.sigmoid(self.alpha * (x - self.beta))

    def trapezoidal(self, x):
        # Eq. 6: μ_T(h) = max(min((h-a)/(b-a), 1, (d-h)/(d-c)), 0)
        eps = 1e-8
        rise = (x - self.a) / (self.b - self.a + eps)
        fall = (self.d - x) / (self.d - self.c + eps)
        return torch.clamp(torch.min(torch.min(rise, fall), torch.ones_like(x)), min=0.0)

    def forward(self, h):
        # Eq. 7: G(h) = w_G·μ_G(h) + w_S·μ_S(h) + w_T·μ_T(h) + b
        return (
            self.w_mu * self.gaussian(h)
            + self.w_sigma * self.sigmoid_mf(h)
            + self.w_T * self.trapezoidal(h)
            + self.b_weight
        )
