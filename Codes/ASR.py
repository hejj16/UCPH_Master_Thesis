import torch
import numpy as np
import pyro


def ASR(z_tips, cov_tips, cov_anc, con_tips_anc, decoder):
    with torch.no_grad():
        z_mu = con_tips_anc @ torch.linalg.inv(cov_tips) @ torch.permute(z_tips[:, :, np.newaxis], (1, 0, 2))
        z_mu = torch.squeeze(z_mu, -1).T
        A = decoder(z_mu)
        return A

