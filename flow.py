import torch
import torch.nn as nn
import torch.distributions as dst
from torch.autograd.functional import jacobian


def tanh_derivative(x):
    return 1 - (torch.tanh(x)) ** 2


class PlanarFlow(nn.Module):

    def __init__(self, D):
        super().__init__()
        self.w = nn.Parameter(data=torch.randn(D, 1))
        self.u = nn.Parameter(data=torch.zeros(1, D))
        self.b = nn.Parameter(data=torch.randn(1, 1))
        
#     @property
#     def u_hat(self):
#         w = self.w.detach()
#         w_dot_u = self.u @ w
#         return self.u + (-1 + torch.log(1 + torch.exp(w_dot_u)) - w_dot_u) * (w.T) / (torch.norm(w)) ** 2
        
    def logabsdet(self, z):
        # (n, D) @ (D, 1) => (n, 1)
        # (n, 1) * (1, D) => (n, D)
        # (n, D) @ (D, 1) => (n, 1)
        return torch.log(torch.abs(
            1 + (tanh_derivative(z @ self.w + self.b) * self.w.T) @ self.u.T
        )).view(-1)
        
    def forward(self, z):
        return z + self.u * torch.tanh(z @ self.w + self.b)
    
    
def batch_jacobian(f, x):
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return jacobian(f_sum, x, create_graph=True).permute(1,0,2)


class NormalizingFlow(nn.Module):
    
    def __init__(self, D=2, K=100):
        super().__init__()
        self.μ = nn.Parameter(torch.zeros(D, ))
        self.pre_σ = nn.Parameter(data=torch.zeros(D, ))
        self.q_0 = dst.multivariate_normal.MultivariateNormal(self.μ, torch.eye(D) * torch.exp(self.pre_σ))
        self.layers = [PlanarFlow(D) for _ in range(K)]
        self.transformation = nn.Sequential(*self.layers)
        
    def sample_from_each_layer(self, n):
        with torch.no_grad():
            z_0 = self.q_0.sample(sample_shape=(n, ))
            zs = [z_0]
            z = z_0
            for layer in self.layers:
                z = layer(z)
                zs.append(z)
            return zs
    
    def sample(self, n, efficient):
        z_0 = self.q_0.rsample(sample_shape=(n, ))  # use rsample so that we can train self.μ and self.pre_σ
        if not efficient:
            z_K = self.transformation(z_0)
            logabsdet_of_all_layers = torch.linalg.slogdet(batch_jacobian(self.transformation, z_0))[1]
            return z_K, self.q_0.log_prob(z_0) - logabsdet_of_all_layers
        else:
            z = z_0
            log_prob = self.q_0.log_prob(z_0)
            for layer in self.layers:
                log_prob -= layer.logabsdet(z)
                z = layer(z)
            return z, log_prob
