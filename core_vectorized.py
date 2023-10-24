import torch
import torch.nn as nn
from torch import tanh, log, abs, bmm, sum
from torch.nn.functional import softplus
import torch.distributions as tdist


def dtanh(x):
    """Tanh derivative"""
    return 1 - (tanh(x)) ** 2


def bt(x):
    """Batch transpose, created with the same spirit as bmm for batch matmul"""
    return x.transpose(1, 2)


def m(x):
    """A smooth function with range being (-1, infinity)"""
    return -1 + softplus(x)


class VecPlanarFlow:
    """
    Vectorized planar flow.
    I made it  like a layer, but it's not a nn.Module because I want to be able to pass in custom parameter values.
    """

    def __init__(self, w, u, b):
        self.bs, self.D = w.shape[0], w.shape[1]
        self.w = w.reshape(self.bs, self.D, 1)
        self.u = u.reshape(self.bs, 1, self.D)
        self.b = b.reshape(self.bs, 1, 1)

    def compute_u_hat(self):
        """
        Vectorized logic:
        out = (m(w_dot_u + 0.5413) - w_dot_u) * bt(self.w):   (self.bs, 1, 1) * (self.bs, self.1, D) => (self.bs, 1, D)
        out = out / sum(self.w ** 2, dim=1, keepdim=True):    (self.bs, 1, D) / (self.bs, 1, 1) => (self.bs, 1, D)
        """
        w_dot_u = bmm(self.u, self.w)  # (self.bs, 1, 1)
        u_hat = self.u + (m(w_dot_u + 0.5413) - w_dot_u) * bt(self.w) / sum(self.w ** 2, dim=1, keepdim=True)
        return u_hat

    def __call__(self, z):
        """
        Non-vectorized logic (computing one term in the sum of eq. 13 from paper):
        log(abs(1 + u.T @ (tanh_derivative(w.T @ z + b) * w)))
        with z having shape (D, ), w having shape (D, ), u having shape (D, ), and b having shape (1, ).

        Vectorized logic:
        out = z:                                 (bs, num_samples, D)
        out = bmm(z, w):                         (bs, num_samples, D) bmm (bs, D, 1) => (bs, num_samples, 1)
        out = out + b:                           (bs, num_samples, 1) + (bs, 1, 1)   => (bs, num_samples, 1)
        out = dtanh(out) * bt(w):                (bs, num_samples, 1) * (bs, 1, D)   => (bs, num_samples, D)
        out = bmm(out, bt(u)):                   (bs, num_samples, D) bmm (bs, D, 1) => (bs, num_samples, 1)
        """
        u_hat = self.compute_u_hat()
        bmm_z_and_w_plus_b = bmm(z, self.w) + self.b
        new_z = z + u_hat * tanh(bmm_z_and_w_plus_b)
        logabsdet = log(abs(
            1 + bmm(dtanh(bmm_z_and_w_plus_b) * bt(self.w), bt(u_hat))
        ))[:, :, 0]  # remove last dimension
        return new_z, logabsdet


class VecNormalizingFlow:

    """
    Vectorized normalizing flow.
    Treat this like a distribution.
    """

    def __init__(self, mu0, sigma0, w, u, b):
        """
        mu0: (bs, D)
        sigma0: (bs, D)
        w: (bs, L, D)
        u: (bs, L, D)
        b: (bs, L)
        """

        super().__init__()

        if w is not None:

            assert w.shape[1] == u.shape[1] == b.shape[1]
            L = w.shape[1]

            self.mu0 = mu0
            self.sigma0 = sigma0
            self.layers = [VecPlanarFlow(w[:, l, :], u[:, l, :], b[:, l]) for l in range(L)]

        else:

            self.mu0 = mu0
            self.sigma0 = sigma0
            self.layers = []

    @property
    def q0(self):
        return tdist.Independent(tdist.Normal(loc=self.mu0, scale=self.sigma0), reinterpreted_batch_ndims=1)

    def rsample(self, num_samples):

        q0 = self.q0
        z = q0.rsample(sample_shape=torch.Size([num_samples]))  # (num_samples, bs, D)
        log_prob = q0.log_prob(z)  # (num_samples, bs)

        z = z.transpose(0, 1)  # (bs, num_samples, D)
        log_prob = log_prob.T  # (bs, num_samples)

        for i, layer in enumerate(self.layers):
            new_z, logabsdet = layer(z)
            log_prob -= logabsdet
            z = new_z

        return z, log_prob  # (bs, num_samples, D), # (bs, num_samples)


class UnconditionalNF(nn.Module):

    def __init__(self, D, L):
        super().__init__()
        self.mu0 = nn.Parameter(torch.zeros(1, D, ))
        self.pre_sigma0 = nn.Parameter(torch.zeros(1, D, ))
        self.w = nn.Parameter(torch.randn(1, L, D))
        self.u = nn.Parameter(torch.zeros(1, L, D))  # starting off with identity transforms
        self.b = nn.Parameter(torch.randn(1, L))

    @property
    def sigma0(self):
        return softplus(self.pre_sigma0)

    def rsample(self, num_samples):
        dist = VecNormalizingFlow(self.mu0, self.sigma0, self.w, self.u, self.b)
        z, log_prob = dist.rsample(num_samples)
        return z[0], log_prob[0]  # remove the first batch dimension


class ConditionalNF(nn.Module):

    def __init__(self, x_dim, D, L):
        super().__init__()
        self.D = D
        self.L = L
        self.amortizer_output_dim = 2 * self.D + self.L * (2 * self.D) + self.L
        self.amortizer = nn.Sequential(
            nn.Linear(x_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, self.amortizer_output_dim)
        )

    def rsample(self, x, num_samples_per_x):
        bs = x.shape[0]
        raw_param = self.amortizer(x)  # (bs, 2 * D + L * (2 * D) + L)
        if self.L == 0:
            dist = VecNormalizingFlow(
                mu0=raw_param[:, :self.D].reshape(bs, self.D),
                sigma0=softplus(raw_param[:, self.D:self.D * 2]).reshape(bs, self.D),  # needs to be positive
                w=None, u=None, b=None
            )
        else:
            dist = VecNormalizingFlow(
                mu0=raw_param[:, :self.D].reshape(bs, self.D),
                sigma0=softplus(raw_param[:, self.D:self.D * 2]).reshape(bs, self.D),  # needs to be positive
                w=raw_param[:, self.D * 2:self.D * 2 + self.L * self.D].reshape(bs, self.L, self.D),
                u=raw_param[:, self.D * 2 + self.L * self.D:self.D * 2 + self.L * self.D * 2].reshape(bs, self.L, self.D),
                b=raw_param[:, self.D * 2 + self.L * self.D * 2:self.D * 2 + self.L * self.D * 2 + self.L].reshape(bs, self.L)
            )
        return dist.rsample(num_samples_per_x)
