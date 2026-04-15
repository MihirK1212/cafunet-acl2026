import torch
import torch.nn as nn


class HybridFeatureEnrichment(nn.Module):
    """Learnable gating for blending specialized and general-purpose features (Eq. 9).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, h_specialized: torch.Tensor, g_prior: torch.Tensor):
        alpha = torch.sigmoid(self.gate(torch.cat([h_specialized, g_prior], dim=-1)))
        return (1 - alpha) * h_specialized + alpha * g_prior
