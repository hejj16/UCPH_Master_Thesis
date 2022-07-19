import torch
import pyro
from pyro.distributions import Gamma, Delta


class ImproperUniform(Delta):
    def log_prob(self, X):
        return torch.zeros_like(super().log_prob(X))


class ImproperGamma(Gamma):
    def log_prob(self, X):
        log_p = super().log_prob(X)
        mode = (self.concentration - 1) / self.rate
        mode_log_p = super().log_prob(mode.expand(X.shape))
        log_p[X > mode] = mode_log_p[X > mode]
        return log_p