import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import config_utils
import constants

config = config_utils.load_config()


class FuzzifierModule(nn.Module):
    def __init__(self, embedding_dim):
        super(FuzzifierModule, self).__init__()
        self.embedding_dim = embedding_dim

        # Parameters for Gaussian membership function
        self.gaussian_c = nn.Parameter(torch.zeros(embedding_dim))  # Gaussian centers
        self.gaussian_sigma = nn.Parameter(torch.ones(embedding_dim))  # Gaussian widths

        # Parameters for Sigmoidal membership function
        self.sigmoid_c = nn.Parameter(torch.zeros(embedding_dim))  # Sigmoidal centers
        self.sigmoid_alpha = nn.Parameter(torch.ones(embedding_dim))  # Sigmoidal slopes

        # Learnable weights for combining Gaussian and Sigmoidal outputs
        self.weight_gaussian = nn.Parameter(torch.tensor(0.5))  # Initial weight for Gaussian
        self.weight_sigmoid = nn.Parameter(torch.tensor(0.5))  # Initial weight for Sigmoidal

    def gaussian_membership(self, x, c, sigma):
        """Gaussian membership function with stability fixes."""
        # Clamp sigma to avoid division by zero
        sigma = torch.clamp(sigma, min=1e-6)
        return torch.exp(-((x - c) ** 2) / (2 * sigma ** 2))

    def sigmoidal_membership(self, x, c, alpha):
        """Sigmoidal membership function with stability fixes."""
        # Clamp the input to the exponential to prevent overflow
        alpha = torch.clamp(alpha, min=1e-6, max=1e6)  # Ensure alpha is positive
        z = alpha * (x - c)
        z = torch.clamp(z, min=-50, max=50)  # Prevent overflow in exp
        return 1 / (1 + torch.exp(-z))

    def forward(self, x):
        # Apply Gaussian membership function
        gaussian_output = self.gaussian_membership(x, self.gaussian_c, self.gaussian_sigma)  # Shape: (batch_size, embedding_dim)

        # Apply Sigmoidal membership function
        sigmoid_output = self.sigmoidal_membership(x, self.sigmoid_c, self.sigmoid_alpha)  # Shape: (batch_size, embedding_dim)

        # Normalize weights to ensure they sum to 1
        weight_gaussian = torch.sigmoid(self.weight_gaussian)  # Range: (0, 1)
        weight_sigmoid = 1 - weight_gaussian  # Complementary weight

        # Weighted combination of Gaussian and Sigmoidal outputs
        combined_output = weight_gaussian * gaussian_output + weight_sigmoid * sigmoid_output  # Shape: (batch_size, embedding_dim)

        # Clamp final output to prevent NaN propagation
        combined_output = torch.clamp(combined_output, min=1e-6, max=1 - 1e-6)

        return combined_output

class MultiLabelFuzzyLayer(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(MultiLabelFuzzyLayer, self).__init__()
        self.fuzzy_modules = nn.ModuleList([
            FuzzifierModule(embedding_dim) for _ in range(num_classes)
        ])

    def forward(self, x):
        fuzzy_outputs = [fuzzy_module(x) for fuzzy_module in self.fuzzy_modules]  # List of (B, D)
        stacked = torch.stack(fuzzy_outputs, dim=1)  # Shape: (B, C, D)
        return stacked  # Each class has its own fuzzy view of x


class Defuzzifier(nn.Module):
    def __init__(self, embedding_dim):
        super(Defuzzifier, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):  # x: (B, C, D)
        return self.linear(x).squeeze(-1)  # Output: (B, C)

class FuzzyMembershipNetwork(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(FuzzyMembershipNetwork, self).__init__()

        self.multi_label_fuzzy_layer = MultiLabelFuzzyLayer(embedding_dim=embedding_dim, num_classes=num_classes)
        self.defuzzifier = Defuzzifier(embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor):
        output = self.multi_label_fuzzy_layer(x)  
        output = self.defuzzifier(output)  
        return output