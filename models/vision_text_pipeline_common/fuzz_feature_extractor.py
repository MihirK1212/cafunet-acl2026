import torch
import torch.nn.functional as F
import math

class FuzzyFeatureExtractor(torch.nn.Module):
    def __init__(self, mu_params, sigma_params, trapezoidal_params, weights):
        super().__init__()
        self.mu = mu_params['mu']
        self.sigma = mu_params['sigma']
        self.alpha = sigma_params['alpha']
        self.beta = sigma_params['beta']
        self.a = trapezoidal_params['a']
        self.b = trapezoidal_params['b']
        self.c = trapezoidal_params['c']
        self.d = trapezoidal_params['d']
        self.w_mu = weights['w_mu']
        self.w_sigma = weights['w_sigma']
        self.w_T = weights['w_T']
        self.b_weight = weights['b']

    def gaussian(self, x):
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (self.sigma * torch.sqrt(torch.tensor(2 * torch.pi, device=x.device)))

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-self.alpha * (x - self.beta)))

    def trapezoidal(self, x):
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)

        cond1 = (x <= self.a)
        cond2 = (self.a < x) & (x <= self.b)
        cond3 = (self.b < x) & (x <= self.c)
        cond4 = (self.c < x) & (x <= self.d)
        cond5 = (x > self.d)

        out = torch.where(cond1, zeros,
              torch.where(cond2, (x - self.a) / (self.b - self.a),
              torch.where(cond3, ones,
              torch.where(cond4, (self.d - x) / (self.d - self.c),
                          zeros))))
        return out

    def forward(self, I):
        I_mu = self.gaussian(I)
        I_sigma = self.sigmoid(I)
        I_T = self.trapezoidal(I)

        I_fuzzy = self.w_mu * I_mu + self.w_sigma * I_sigma + self.w_T * I_T + self.b_weight
        return I_fuzzy


# class FuzzyFeatureExtractor(torch.nn.Module):
#     def __init__(self, mu_params, sigma_params, trapezoidal_params, weights):
#         super().__init__()
#         self.mu = torch.nn.Parameter(torch.tensor(mu_params['mu'], dtype=torch.float32))
#         self.sigma = torch.nn.Parameter(torch.tensor(mu_params['sigma'], dtype=torch.float32))

#         self.alpha = torch.nn.Parameter(torch.tensor(sigma_params['alpha'], dtype=torch.float32))
#         self.beta = torch.nn.Parameter(torch.tensor(sigma_params['beta'], dtype=torch.float32))

#         self.a = torch.nn.Parameter(torch.tensor(trapezoidal_params['a'], dtype=torch.float32))
#         self.b = torch.nn.Parameter(torch.tensor(trapezoidal_params['b'], dtype=torch.float32))
#         self.c = torch.nn.Parameter(torch.tensor(trapezoidal_params['c'], dtype=torch.float32))
#         self.d = torch.nn.Parameter(torch.tensor(trapezoidal_params['d'], dtype=torch.float32))

#         self.w_mu = torch.nn.Parameter(torch.tensor(weights['w_mu'], dtype=torch.float32))
#         self.w_sigma = torch.nn.Parameter(torch.tensor(weights['w_sigma'], dtype=torch.float32))
#         self.w_T = torch.nn.Parameter(torch.tensor(weights['w_T'], dtype=torch.float32))
#         self.b_weight = torch.nn.Parameter(torch.tensor(weights['b'], dtype=torch.float32))

#     def gaussian(self, x):
#         return torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (self.sigma * math.sqrt(2 * math.pi))

#     def sigmoid(self, x):
#         return 1 / (1 + torch.exp(-self.alpha * (x - self.beta)))

#     def trapezoidal(self, x):
#         zeros = torch.zeros_like(x)
#         ones = torch.ones_like(x)

#         cond1 = (x <= self.a)
#         cond2 = (self.a < x) & (x <= self.b)
#         cond3 = (self.b < x) & (x <= self.c)
#         cond4 = (self.c < x) & (x <= self.d)
#         cond5 = (x > self.d)

#         out = torch.where(cond1, zeros,
#               torch.where(cond2, (x - self.a) / (self.b - self.a),
#               torch.where(cond3, ones,
#               torch.where(cond4, (self.d - x) / (self.d - self.c),
#                           zeros))))
#         return out

#     def forward(self, I):
#         I_mu = self.gaussian(I)
#         I_sigma = self.sigmoid(I)
#         I_T = self.trapezoidal(I)

#         I_fuzzy = self.w_mu * I_mu + self.w_sigma * I_sigma + self.w_T * I_T + self.b_weight
#         return I_fuzzy
