import torch
import torch.nn as nn
import torch.distributions as pdist
from torch.autograd.functional import jacobian
m = lambda x : -1 + nn.functional.softplus(x)
from torch import log, abs, sum


def tanh_derivative(x):
    return 1 - (torch.tanh(x)) ** 2


class PlanarFlow(nn.Module):

    def __init__(self, D):
        super().__init__()
        self.w = nn.Parameter(data=torch.randn(D, 1))
        self.u = nn.Parameter(data=torch.zeros(1, D))
        self.b = nn.Parameter(data=torch.randn(1, 1))

    @staticmethod
    def logabsdet(z, w, u, b):
        return log(abs(1 + (tanh_derivative(z @ w + b) * w.T) @ u.T)).view(-1)

    def forward(self, z):
        w_dot_u = (self.u @ self.w).squeeze()
        u_hat = self.u + (m(w_dot_u + 0.5413) - w_dot_u) * self.w.T / sum(self.w ** 2)
        return z + u_hat * torch.tanh(z @ self.w + self.b), self.logabsdet(z, self.w, u_hat, self.b)
    
    
def batch_jacobian(f, x):
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return jacobian(f_sum, x, create_graph=True).permute(1,0,2)


class DiagonalGaussian(nn.Module):
    
    def __init__(self, D):
        super().__init__()
        self.μ = nn.Parameter(torch.zeros(D, ))
        self.pre_σ = nn.Parameter(data=torch.zeros(D, ))
        
    def sample(self, n):
        dist = pdist.multivariate_normal.MultivariateNormal(self.μ, torch.eye(D) * torch.exp(self.pre_σ))
        samples = dist.rsample(sample_shape=(n, ))
        return samples, dist.log_prob(samples)
    
    
class FullRankGaussian(nn.Module):
    
    def __init__(self, D):
        super().__init__()
        self.μ = nn.Parameter(torch.zeros(D, ))
        self.cov_decomp = nn.Parameter(torch.linalg.cholesky(torch.eye(D), upper=True))
        
    @property
    def cov(self):
        temp = torch.triu(self.cov_decomp)
        return temp.T @ temp
        
    def sample(self, n):
        dist = pdist.multivariate_normal.MultivariateNormal(self.μ, self.cov)
        samples = dist.rsample(sample_shape=(n, ))
        return samples, dist.log_prob(samples)
    
        
class NormalizingFlow(nn.Module):
    
    def __init__(self, D, K):
        super().__init__()
        self.D = D
        self.μ = nn.Parameter(torch.zeros(D, ))
        self.pre_σ = nn.Parameter(data=torch.zeros(D, ))
        self.layers = [PlanarFlow(D) for _ in range(K)]
        self.transformation = nn.Sequential(*self.layers)

    @property
    def q_0(self):
        return pdist.multivariate_normal.MultivariateNormal(self.μ, torch.eye(self.D) * nn.functional.softplus(self.pre_σ))
        
    def sample_from_each_layer(self, n):
        with torch.no_grad():
            z_0 = self.q_0.sample(sample_shape=(n, ))
            zs = [z_0]
            z = z_0
            for layer in self.layers:
                z, _ = layer(z)
                zs.append(z)
            return zs
    
    def sample(self, n, efficient=True):
        z_0 = self.q_0.rsample(sample_shape=(n, ))  # use rsample so that we can train self.μ and self.pre_σ
        if not efficient:
            z_K = self.transformation(z_0)
            logabsdet_of_all_layers = torch.linalg.slogdet(batch_jacobian(self.transformation, z_0))[1]
            return z_K, self.q_0.log_prob(z_0) - logabsdet_of_all_layers
        else:
            z = z_0
            log_prob = self.q_0.log_prob(z_0)
            for layer in self.layers:
                new_z, term = layer(z)
                log_prob -= term
                z = new_z
            return z, log_prob
