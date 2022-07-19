import pyro
import torch
import pyro.distributions as dist
from torch.distributions import constraints
from Networks_ARVAE import *
from JAXFunctions import *
from SelfDefinedDistributions import *
from torch import nn
import numpy as np
from helper import load_blosum


class VAE(nn.Module):
    def __init__(self,
                 n_z,
                 n_c,
                 n_l,
                 device="cuda"):
        super().__init__()
        self.n_z = n_z
        self.n_c = n_c
        self.n_l = n_l
        self.device = device
        self.B = load_blosum(n_c=n_c).to(self.device)
        self.nw_fn = jax.jit(nw(gap=-4, temp=1)[0])
        self.encoder = Encoder_ARVAE(n_l=n_l,
                                     n_z=n_z,
                                     n_c=n_c)

        self.decoder = Decoder_ARVAE(n_l=n_l,
                                     n_z=n_z,
                                     n_c=n_c,
                                     min_deconv_dim=42,
                                     dropout=0.45,
                                     gru_hidden_size=512,
                                     gru_num_layer=1)

        self.to(self.device)
        self.encoder.train()
        self.decoder.train()

    def iid_model(self,
                  aligned_S,
                  num_batch):
        pyro.module("decoder", self.decoder)

        # sample z
        with pyro.poutine.scale(None, num_batch):
            with pyro.plate("sequences_obs", aligned_S.shape[0], dim=-2):
                with pyro.plate("dim", self.n_z, dim=-1):
                    Z = pyro.sample("latent",
                                    dist.Normal(torch.zeros(aligned_S.shape[0],  self.n_z, device=self.device),
                                                torch.ones(aligned_S.shape[0],  self.n_z, device=self.device)))
                S_onehot = torch.zeros_like(aligned_S[:, :, None]).expand([-1, -1, self.n_c]).clone().scatter_(-1, aligned_S[:, :, None], 1)
                S_onehot = S_onehot[:, :, :].float().to(self.device)
                A = self.decoder(S_onehot, Z, True)
                with pyro.plate("seq_length", aligned_S.shape[1], dim=-1):
                    pyro.sample("obs", dist.Categorical(logits=A), obs=aligned_S)

    def iid_guide(self,
                  aligned_S,
                  num_batch):

        pyro.module("encoder", self.encoder)
        S_onehot = torch.zeros_like(aligned_S[:, :, None]).expand([-1, -1, self.n_c]).clone().scatter_(-1, aligned_S[:, :, None], 1)
        S_onehot = S_onehot[:, :, :].float().to(self.device)

        Z_loc, Z_scale = self.encoder(S_onehot, None, return_scale=True)
        with pyro.poutine.scale(None, num_batch):
            with pyro.plate("sequences_obs", aligned_S.shape[0], dim=-2):
                with pyro.plate("dim", self.n_z, dim=-1):
                    pyro.sample("latent", dist.Normal(Z_loc, Z_scale))

    def standard_asr_model(self,
                           aligned_S,
                           batch_distance):
        pyro.module("decoder", self.decoder)

        # sample parameters of OU hyper-priors
        with pyro.plate("hyper_alpha", 3):
            alpha = pyro.sample("alpha", dist.Gamma(torch.ones(1, device=self.device) * 2,
                                                    torch.ones(1, device=self.device)).expand([3]))
        # sample from OU hyper-priors
        with pyro.plate("sigma_lambda", self.n_z):
            sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand([self.n_z]))
            sigma_n = pyro.sample("sigma_n", dist.HalfNormal(alpha[1]).expand([self.n_z]))
            lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand([self.n_z]))

        cov = batch_distance[np.newaxis, :, :].expand(
            [self.n_z, batch_distance.shape[0], batch_distance.shape[0]]) / lambd[:, np.newaxis, np.newaxis]
        cov = (torch.exp(-cov) * sigma_f[:, np.newaxis, np.newaxis] ** 2 +
               torch.eye(cov.shape[1], device=self.device)[np.newaxis, :, :] * sigma_n[:, np.newaxis, np.newaxis] ** 2)

        # sample z
        with pyro.plate("dim", self.n_z, dim=-1):
            Z = pyro.sample("latent", dist.MultivariateNormal(
                torch.zeros(self.n_z, aligned_S.shape[0], device=self.device), cov))
        Z = Z.T

        S_onehot = torch.zeros_like(aligned_S[:, :, None]).expand([-1, -1, self.n_c]).clone().scatter_(-1,
                                                                                                       aligned_S[:, :,
                                                                                                       None], 1)
        S_onehot = S_onehot[:, :, :].float().to(self.device)
        A = self.decoder(S_onehot, Z, True)
        with pyro.plate("sequences_obs", aligned_S.shape[0], dim=-2):
            with pyro.plate("seq_length", aligned_S.shape[1], dim=-1):
                pyro.sample("obs", dist.Categorical(logits=A), obs=aligned_S)

    def standard_asr_guide(self,
                           aligned_S,
                           batch_distance):
        pyro.module("encoder", self.encoder)

        alpha_loc = pyro.param("alpha_loc", torch.ones(3, device=self.device),
                               constraint=constraints.interval(0.1, 3))
        sigma_f_loc = pyro.param("sigma_f_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                                 constraint=constraints.interval(0.1, 3))
        sigma_n_loc = pyro.param("sigma_n_loc", torch.ones(self.n_z, device=self.device) * 0.2,
                                 constraint=constraints.interval(0.1, 3))
        lambd_loc = pyro.param("lambd_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                               constraint=constraints.interval(0.1, 3))

        with pyro.plate("hyper_alpha", 3):
            pyro.sample("alpha", dist.Delta(alpha_loc))
        with pyro.plate("sigma_lambda", self.n_z):
            pyro.sample("sigma_f", dist.Delta(sigma_f_loc))
            pyro.sample("sigma_n", dist.Delta(sigma_n_loc))
            pyro.sample("lambd", dist.Delta(lambd_loc))

        S_onehot = torch.zeros_like(aligned_S[:, :, None]).expand([-1, -1, self.n_c]).clone().scatter_(-1,
                                                                                                       aligned_S[:, :,
                                                                                                       None], 1)
        S_onehot = S_onehot[:, :, :].float().to(self.device)

        Z_loc, Z_scale = self.encoder(S_onehot, None, return_scale=True)
        with pyro.plate("dim", self.n_z, dim=-1):
            pyro.sample("latent", dist.Normal(Z_loc.T, Z_scale.T).to_event(1))

    def standard_asr_model_latent_tree(self,
                                       aligned_S,
                                       tree_emb_dim=40):
        self.tree_emb_dim = tree_emb_dim
        pyro.module("decoder", self.decoder)
        tree_emb_loc = pyro.param("tree_embed_loc",
                                  torch.randn(aligned_S.shape[0], tree_emb_dim, device=self.device))

        # sample parameters of OU hyper-priors
        with pyro.plate("hyper_alpha", 3):
            alpha = pyro.sample("alpha", dist.Gamma(torch.ones(1, device=self.device) * 2,
                                                    torch.ones(1, device=self.device)).expand([3]))
        # sample from OU hyper-priors
        with pyro.plate("sigma_lambda", self.n_z):
            sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand([self.n_z]))
            sigma_n = pyro.sample("sigma_n", dist.HalfNormal(alpha[1]).expand([self.n_z]))
            lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand([self.n_z]))

        with pyro.plate("tree_num", aligned_S.shape[0], dim=-2):
            with pyro.plate("tree_dim", tree_emb_dim, dim=-1):
                tree_emb_len = pyro.sample("tree_emb_len_beta", dist.Beta(torch.tensor([5.55], device=self.device),
                                                                          torch.tensor([2.28], device=self.device))
                                           )
        tree_embedding = tree_emb_loc / tree_emb_loc.norm(dim=-1, keepdim=True) * tree_emb_len
        batch_distance = self._get_D(tree_embedding, 1e-5)

        cov = batch_distance[np.newaxis, :, :].expand(
            [self.n_z, batch_distance.shape[0], batch_distance.shape[0]]) / lambd[:, np.newaxis, np.newaxis]
        cov = (torch.exp(-cov) * sigma_f[:, np.newaxis, np.newaxis] ** 2 +
               torch.eye(cov.shape[1], device=self.device)[np.newaxis, :, :] * sigma_n[:, np.newaxis, np.newaxis] ** 2)

        # sample z
        with pyro.plate("dim", self.n_z, dim=-1):
            Z = pyro.sample("latent", dist.MultivariateNormal(
                torch.zeros(self.n_z, aligned_S.shape[0], device=self.device), cov))
        Z = Z.T

        S_onehot = torch.zeros_like(aligned_S[:, :, None]).expand([-1, -1, self.n_c]).clone().scatter_(-1,
                                                                                                       aligned_S[:, :,
                                                                                                       None], 1)
        S_onehot = S_onehot[:, :, :].float().to(self.device)
        A = self.decoder(S_onehot, Z, True)
        with pyro.plate("sequences_obs", aligned_S.shape[0], dim=-2):
            with pyro.plate("seq_length", aligned_S.shape[1], dim=-1):
                pyro.sample("obs", dist.Categorical(logits=A), obs=aligned_S)

    def standard_asr_guide_latent_tree(self,
                                       aligned_S,
                                       tree_emb_dim=40):
        pyro.module("encoder", self.encoder)

        alpha_loc = pyro.param("alpha_loc", torch.ones(3, device=self.device),
                               constraint=constraints.interval(0.1, 3))
        sigma_f_loc = pyro.param("sigma_f_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                                 constraint=constraints.interval(0.1, 3))
        sigma_n_loc = pyro.param("sigma_n_loc", torch.ones(self.n_z, device=self.device) * 0.2,
                                 constraint=constraints.interval(0.1, 3))
        lambd_loc = pyro.param("lambd_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                               constraint=constraints.interval(0.1, 3))

        tree_emb_len = pyro.param("tree_embed_len",
                                  torch.ones(aligned_S.shape[0], 1, device=self.device) * 0.5,
                                  constraint=constraints.interval(1e-3, 1 - 1e-3))
        with pyro.plate("tree_num", aligned_S.shape[0], dim=-2):
            with pyro.plate("tree_dim", tree_emb_dim, dim=-1):
                pyro.sample("tree_emb_len_beta", dist.Delta(tree_emb_len))

        with pyro.plate("hyper_alpha", 3):
            pyro.sample("alpha", dist.Delta(alpha_loc))
        with pyro.plate("sigma_lambda", self.n_z):
            pyro.sample("sigma_f", dist.Delta(sigma_f_loc))
            pyro.sample("sigma_n", dist.Delta(sigma_n_loc))
            pyro.sample("lambd", dist.Delta(lambd_loc))

        S_onehot = torch.zeros_like(aligned_S[:, :, None]).expand([-1, -1, self.n_c]).clone().scatter_(-1,
                                                                                                       aligned_S[:, :,
                                                                                                       None], 1)
        S_onehot = S_onehot[:, :, :].float().to(self.device)

        Z_loc, Z_scale = self.encoder(S_onehot, None, return_scale=True)
        with pyro.plate("dim", self.n_z, dim=-1):
            pyro.sample("latent", dist.Normal(Z_loc.T, Z_scale.T).to_event(1))

    def standard_asr_batched_model(self,
                                   aligned_S,
                                   aligned_S_back,
                                   batch_distance,
                                   num_batch):
        pyro.module("decoder", self.decoder)

        # sample parameters of OU hyper-priors
        with pyro.plate("hyper_alpha", 3):
            alpha = pyro.sample("alpha", dist.Gamma(torch.ones(1, device=self.device) * 2,
                                                    torch.ones(1, device=self.device)).expand([3]))
        # sample from OU hyper-priors
        with pyro.plate("sigma_lambda", self.n_z):
            sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand([self.n_z]))
            sigma_n = pyro.sample("sigma_n", dist.HalfNormal(alpha[1]).expand([self.n_z]))
            lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand([self.n_z]))

        assert batch_distance.shape[0] == aligned_S_back.shape[0] + aligned_S.shape[0]

        cov = batch_distance[np.newaxis, :, :].expand(
            [self.n_z, batch_distance.shape[0], batch_distance.shape[0]]) / lambd[:, np.newaxis, np.newaxis]
        cov = (torch.exp(-cov) * sigma_f[:, np.newaxis, np.newaxis] ** 2 +
               torch.eye(cov.shape[1], device=self.device)[np.newaxis, :, :] * sigma_n[:, np.newaxis, np.newaxis] ** 2)

        cov_back = cov[:, :aligned_S_back.shape[0], :aligned_S_back.shape[0]]

        # sample z
        with pyro.plate("dim_back", self.n_z, dim=-1):
            Z_back = pyro.sample("latent_back", dist.MultivariateNormal(
                torch.zeros(self.n_z, aligned_S_back.shape[0], device=self.device), cov_back))
        Z_back = Z_back.T

        S_onehot_back = torch.zeros_like(
            aligned_S_back[:, :, None]
        ).expand([-1, -1, self.n_c]).clone().scatter_(-1, aligned_S_back[:, :, None], 1)
        S_onehot_back = S_onehot_back[:, :, :].float().to(self.device)
        A_back = self.decoder(S_onehot_back, Z_back, True)
        with pyro.plate("sequences_obs_back", aligned_S_back.shape[0], dim=-2):
            with pyro.plate("seq_length_back", aligned_S_back.shape[1], dim=-1):
                pyro.sample("obs_back", dist.Categorical(logits=A_back), obs=aligned_S_back)

        cov_batch = cov[:, aligned_S_back.shape[0]:, aligned_S_back.shape[0]:]
        cov_batch_back = cov[:, aligned_S_back.shape[0]:, :aligned_S_back.shape[0]]
        cov_back_inv = torch.linalg.inv(cov_back)
        mu_batch = (cov_batch_back @ cov_back_inv @ Z_back.T[:, :, None]).squeeze(-1)
        # n_z, N_batch, N_back @ n_z, N_back, N_back @ n_z, N_back, 1
        cov_batch = cov_batch - cov_batch_back @ cov_back_inv @ cov_batch_back.transpose(1, 2)

        with pyro.poutine.scale(None, num_batch):
            # sample z
            with pyro.plate("dim_batch", self.n_z, dim=-1):
                Z_batch = pyro.sample("latent_batch", dist.MultivariateNormal(mu_batch, cov_batch))
            Z_batch = Z_batch.T

            S_onehot_batch = torch.zeros_like(
                aligned_S[:, :, None]
            ).expand([-1, -1, self.n_c]).clone().scatter_(-1, aligned_S[:, :, None], 1)
            S_onehot_batch = S_onehot_batch[:, :, :].float().to(self.device)
            A_batch = self.decoder(S_onehot_batch, Z_batch, True)
            with pyro.plate("sequences_obs_batch", aligned_S.shape[0], dim=-2):
                with pyro.plate("seq_length_batch", aligned_S.shape[1], dim=-1):
                    pyro.sample("obs_batch", dist.Categorical(logits=A_batch), obs=aligned_S)

    def standard_asr_batched_guide(self,
                                   aligned_S,
                                   aligned_S_back,
                                   batch_distance,
                                   num_batch):
        pyro.module("encoder", self.encoder)

        alpha_loc = pyro.param("alpha_loc", torch.ones(3, device=self.device),
                               constraint=constraints.interval(0.1, 3))
        sigma_f_loc = pyro.param("sigma_f_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                                 constraint=constraints.interval(0.1, 3))
        sigma_n_loc = pyro.param("sigma_n_loc", torch.ones(self.n_z, device=self.device) * 0.2,
                                 constraint=constraints.interval(0.1, 3))
        lambd_loc = pyro.param("lambd_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                               constraint=constraints.interval(0.1, 3))

        with pyro.plate("hyper_alpha", 3):
            pyro.sample("alpha", dist.Delta(alpha_loc))
        with pyro.plate("sigma_lambda", self.n_z):
            pyro.sample("sigma_f", dist.Delta(sigma_f_loc))
            pyro.sample("sigma_n", dist.Delta(sigma_n_loc))
            pyro.sample("lambd", dist.Delta(lambd_loc))

        S_onehot_back = torch.zeros_like(
            aligned_S_back[:, :, None]
        ).expand([-1, -1, self.n_c]).clone().scatter_(-1, aligned_S_back[:, :, None], 1)
        S_onehot_back = S_onehot_back[:, :, :].float().to(self.device)

        Z_loc_back, Z_scale_back = self.encoder(S_onehot_back, None, return_scale=True)
        with pyro.plate("dim_back", self.n_z, dim=-1):
            pyro.sample("latent_back", dist.Normal(Z_loc_back.T, Z_scale_back.T).to_event(1))

        with pyro.poutine.scale(None, num_batch):
            S_onehot_batch = torch.zeros_like(
                aligned_S[:, :, None]
            ).expand([-1, -1, self.n_c]).clone().scatter_(-1, aligned_S[:, :, None], 1)
            S_onehot_batch = S_onehot_batch[:, :, :].float().to(self.device)

            Z_loc_batch, Z_scale_batch = self.encoder(S_onehot_batch, None, return_scale=True)
            with pyro.plate("dim_batch", self.n_z, dim=-1):
                pyro.sample("latent_batch", dist.Normal(Z_loc_batch.T, Z_scale_batch.T).to_event(1))

    def standard_asr_batched_model_latent_tree(self,
                                               aligned_S,
                                               S_length,
                                               S_index,
                                               num_batch,
                                               aligned_S_back,
                                               S_back_length,
                                               S_back_index,
                                               num_instance,
                                               tree_emb_dim=40, ):
        self.tree_emb_dim = tree_emb_dim
        tree_emb_loc = pyro.param("tree_embed_loc",
                                  torch.randn(num_instance, tree_emb_dim, device=self.device))

        tree_emb_loc_back = tree_emb_loc[S_back_index]
        with pyro.plate("tree_num_back", aligned_S_back.shape[0], dim=-2):
            with pyro.plate("tree_dim_back", tree_emb_dim, dim=-1):
                tree_emb_len_back = pyro.sample("tree_emb_len_beta_back", dist.Beta(torch.tensor([5.55], device=self.device),
                                                                               torch.tensor([2.28], device=self.device))
                                           )
        tree_embedding_back = tree_emb_loc_back / tree_emb_loc_back.norm(dim=-1, keepdim=True) * tree_emb_len_back

        with pyro.poutine.scale(None, num_batch):
            tree_emb_loc_batch = tree_emb_loc[S_index]
            with pyro.plate("tree_num_batch", aligned_S.shape[0], dim=-2):
                with pyro.plate("tree_dim_batch", tree_emb_dim, dim=-1):
                    tree_emb_len_batch = pyro.sample("tree_emb_len_beta_batch",
                                               dist.Beta(torch.tensor([5.55], device=self.device),
                                                         torch.tensor([2.28], device=self.device))
                                               )
            tree_embedding_batch = tree_emb_loc_batch / tree_emb_loc_batch.norm(dim=-1, keepdim=True) * tree_emb_len_batch
        tree_embedding = torch.cat([tree_embedding_back, tree_embedding_batch], dim=0)
        batch_distance = self._get_D(tree_embedding, 1e-5)
        self.standard_asr_batched_model(aligned_S,
                                        aligned_S_back,
                                        batch_distance,
                                        num_batch)

    def standard_asr_batched_guide_latent_tree(self,
                                               aligned_S,
                                               S_length,
                                               S_index,
                                               num_batch,
                                               aligned_S_back,
                                               S_back_length,
                                               S_back_index,
                                               num_instance,
                                               tree_emb_dim=40, ):
        tree_emb_len = pyro.param("tree_embed_len",
                                  torch.ones(num_instance, 1, device=self.device) * 0.5,
                                  constraint=constraints.interval(1e-3, 1 - 1e-3))

        tree_emb_len_back = tree_emb_len[S_back_index]
        with pyro.plate("tree_num_back", aligned_S_back.shape[0], dim=-2):
            with pyro.plate("tree_dim_back", tree_emb_dim, dim=-1):
                pyro.sample("tree_emb_len_beta_back", dist.Delta(tree_emb_len_back))

        tree_emb_len_batch = tree_emb_len[S_index]
        with pyro.poutine.scale(None, num_batch):
            with pyro.plate("tree_num_batch", aligned_S.shape[0], dim=-2):
                with pyro.plate("tree_dim_batch", tree_emb_dim, dim=-1):
                    pyro.sample("tree_emb_len_beta_batch", dist.Delta(tree_emb_len_batch))
        self.standard_asr_batched_guide(aligned_S,
                                        aligned_S_back,
                                        None,
                                        num_batch)

    def unbatch_model(self,
                      Unaligned_S,
                      S_length,
                      msa_ref,
                      batch_distance,
                      S_ave_pos):
        pyro.module("decoder", self.decoder)

        # sample parameters of OU hyper-priors
        with pyro.plate("hyper_alpha", 3):
            alpha = pyro.sample("alpha", dist.Gamma(torch.ones(1, device=self.device) * 2,
                                                    torch.ones(1, device=self.device)).expand([3]))
        # sample from OU hyper-priors
        with pyro.plate("sigma_lambda", self.n_z):
            sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand([self.n_z]))
            sigma_n = pyro.sample("sigma_n", dist.HalfNormal(alpha[1]).expand([self.n_z]))
            lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand([self.n_z]))

        cov = batch_distance[np.newaxis, :, :].expand(
            [self.n_z, batch_distance.shape[0], batch_distance.shape[0]]) / lambd[:, np.newaxis, np.newaxis]
        cov = (torch.exp(-cov) * sigma_f[:, np.newaxis, np.newaxis] ** 2 +
               torch.eye(cov.shape[1], device=self.device)[np.newaxis, :, :] * sigma_n[:, np.newaxis, np.newaxis] ** 2)

        # sample z
        with pyro.plate("dim", self.n_z, dim=-1):
            Z = pyro.sample("latent", dist.MultivariateNormal(
                torch.zeros(self.n_z, Unaligned_S.shape[0], device=self.device), cov))
        Z = Z.T

        with pyro.plate("sequences_path", Unaligned_S.shape[0], dim=-1):
            path = pyro.sample("path", ImproperUniform(torch.zeros(1,
                                                                   self.n_l,
                                                                   Unaligned_S.shape[1],
                                                                   device=self.device)).to_event(2))
        mask = (torch.arange(Unaligned_S.shape[1])[None, :] < torch.tensor(S_length)[:, None]).to(self.device)
        with pyro.plate("sequences_obs", Unaligned_S.shape[0], dim=-2):
            with pyro.plate("teacher_length", self.n_l, dim=-1):
                teachor_S = pyro.sample("teacher_S",
                                        ImproperUniform(torch.ones(Unaligned_S.shape[0],
                                                                   self.n_l,
                                                                   self.n_c,
                                                                   device=self.device)).to_event(1))
            loc = self.decoder(teachor_S, Z, True)
            with pyro.plate("msa_length", self.n_l, dim=-1):
                A = pyro.sample("S", dist.RelaxedOneHotCategoricalStraightThrough(torch.ones(1, device=self.device),
                                                                                  logits=loc))
            Unaligned_A = path.permute((0, 2, 1)) @ loc
            with pyro.plate("seq_length", Unaligned_S.shape[1], dim=-1):
                pyro.sample("obs",
                            dist.Categorical(logits=Unaligned_A).mask(mask),
                            obs=Unaligned_S)

    def unbatch_guide(self,
                      Unaligned_S,
                      S_length,
                      msa_ref,
                      batch_distance,
                      S_ave_pos):
        pyro.module("encoder", self.encoder)

        blosum = pyro.param("Blosum", self.B[:-1, :-1])

        alpha_loc = pyro.param("alpha_loc", torch.ones(3, device=self.device),
                               constraint=constraints.interval(0.1, 3))
        sigma_f_loc = pyro.param("sigma_f_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                                 constraint=constraints.interval(0.1, 3))
        sigma_n_loc = pyro.param("sigma_n_loc", torch.ones(self.n_z, device=self.device) * 0.2,
                                 constraint=constraints.interval(0.1, 3))
        lambd_loc = pyro.param("lambd_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                               constraint=constraints.interval(0.1, 3))

        with pyro.plate("hyper_alpha", 3):
            pyro.sample("alpha", dist.Delta(alpha_loc))
        with pyro.plate("sigma_lambda", self.n_z):
            pyro.sample("sigma_f", dist.Delta(sigma_f_loc))
            pyro.sample("sigma_n", dist.Delta(sigma_n_loc))
            pyro.sample("lambd", dist.Delta(lambd_loc))

        S_onehot = torch.zeros_like(Unaligned_S[:, :, None]).expand([-1, -1, self.n_c]).clone().scatter_(-1, Unaligned_S[:, :, None], 1)
        S_onehot = S_onehot[:, :, :].float().to(self.device)
        ref_emb = msa_ref
        mask = (torch.arange(Unaligned_S.shape[1])[None, :] < torch.tensor(S_length)[:, None]).to(self.device)
        s_emb = (S_onehot * mask[:, :, None])[:, :, :-1]

        similar_tensor = ref_emb @ blosum @ s_emb.transpose(1, 2)

        path = snw(similar_tensor, list(S_length), self.nw_fn)

        with pyro.plate("sequences_path", S_onehot.shape[0]):
            pyro.sample("path", dist.Delta(path).to_event(2))

        S = path @ S_onehot  # N, L, l @ N, l, n_c
        assert torch.all(S[:, :, -1] == 0)
        S[:, :, -1] = S[:, :, -1] + 1 - path.sum(-1)  # fill the gaps with onehot-encoding of "-"

        S_ave = torch.sum(S.detach(), dim=0, keepdim=True)[:, :, :-1]
        S_ave_pos.append(S_ave)

        Z_loc, Z_scale = self.encoder(S, None, return_scale=True)
        with pyro.plate("sequences_obs", S.shape[0], dim=-2):
            with pyro.plate("teacher_length", self.n_l, dim=-1):
                pyro.sample("teacher_S", Delta(S).to_event(1))
            with pyro.plate("msa_length", self.n_l, dim=-1):
                S = S + 1e-4
                S = S / S.sum(-1, True)
                pyro.sample("S", Delta(S).to_event(1))

        with pyro.plate("dim", self.n_z, dim=-1):
            pyro.sample("latent", dist.Normal(Z_loc.T, Z_scale.T).to_event(1))

    def unbatch_model_latent_tree(self,
                                  Unaligned_S,
                                  S_length,
                                  msa_ref,
                                  S_ave_pos,
                                  tree_emb_dim=40,
                                  regularization_D=False):
        self.tree_emb_dim = tree_emb_dim
        pyro.module("decoder", self.decoder)

        tree_emb_loc = pyro.param("tree_embed_loc",
                                  torch.randn(Unaligned_S.shape[0], tree_emb_dim, device=self.device))

        # sample parameters of OU hyper-priors
        with pyro.plate("hyper_alpha", 3):
            alpha = pyro.sample("alpha", dist.Gamma(torch.ones(1, device=self.device) * 2,
                                                    torch.ones(1, device=self.device)).expand([3]))
        # sample from OU hyper-priors
        with pyro.plate("sigma_lambda", self.n_z):
            sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand([self.n_z]))
            sigma_n = pyro.sample("sigma_n", dist.HalfNormal(alpha[1]).expand([self.n_z]))
            lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand([self.n_z]))

        with pyro.plate("tree_num", Unaligned_S.shape[0], dim=-2):
            with pyro.plate("tree_dim", tree_emb_dim, dim=-1):
                tree_emb_len = pyro.sample("tree_emb_len_beta", dist.Beta(torch.tensor([5.55], device=self.device),
                                                                          torch.tensor([2.28], device=self.device))
                                           )
        tree_embedding = tree_emb_loc / tree_emb_loc.norm(dim=-1, keepdim=True) * tree_emb_len
        batch_distance = self._get_D(tree_embedding, 1e-5)

        if regularization_D:
            gamma_beta = torch.tensor(1.5, device=self.device)
            gamma_alpha = torch.tensor(3.0, device=self.device)
            with pyro.plate("D_values", (batch_distance.shape[0] - 1) * batch_distance.shape[0] / 2):
                pyro.sample("D",
                            ImproperGamma(concentration=gamma_alpha, rate=gamma_beta),
                            obs=batch_distance[
                                torch.triu_indices(batch_distance.shape[0], batch_distance.shape[0], 1)[0],
                                torch.triu_indices(batch_distance.shape[0], batch_distance.shape[0], 1)[1]])

        cov = batch_distance[np.newaxis, :, :].expand(
            [self.n_z, batch_distance.shape[0], batch_distance.shape[0]]) / lambd[:, np.newaxis, np.newaxis]
        cov = (torch.exp(-cov) * sigma_f[:, np.newaxis, np.newaxis] ** 2 +
               torch.eye(cov.shape[1], device=self.device)[np.newaxis, :, :] * sigma_n[:, np.newaxis, np.newaxis] ** 2)

        # sample z
        with pyro.plate("dim", self.n_z, dim=-1):
            Z = pyro.sample("latent", dist.MultivariateNormal(
                torch.zeros(self.n_z, Unaligned_S.shape[0], device=self.device), cov))
        Z = Z.T

        with pyro.plate("sequences_path", Unaligned_S.shape[0], dim=-1):
            path = pyro.sample("path", ImproperUniform(torch.zeros(1,
                                                                   self.n_l,
                                                                   Unaligned_S.shape[1],
                                                                   device=self.device)).to_event(2))
        mask = (torch.arange(Unaligned_S.shape[1])[None, :] < torch.tensor(S_length)[:, None]).to(self.device)
        with pyro.plate("sequences_obs", Unaligned_S.shape[0], dim=-2):
            with pyro.plate("teacher_length", self.n_l, dim=-1):
                teachor_S = pyro.sample("teacher_S",
                                        ImproperUniform(torch.ones(Unaligned_S.shape[0],
                                                                   self.n_l,
                                                                   self.n_c,
                                                                   device=self.device)).to_event(1))
            loc = self.decoder(teachor_S, Z, True)
            with pyro.plate("msa_length", self.n_l, dim=-1):
                A = pyro.sample("S", dist.RelaxedOneHotCategoricalStraightThrough(torch.ones(1, device=self.device),
                                                                                  logits=loc))
            Unaligned_A = path.permute((0, 2, 1)) @ loc
            with pyro.plate("seq_length", Unaligned_S.shape[1], dim=-1):
                pyro.sample("obs",
                            dist.Categorical(logits=Unaligned_A).mask(mask),
                            obs=Unaligned_S)

    def unbatch_guide_latent_tree(self,
                                  Unaligned_S,
                                  S_length,
                                  msa_ref,
                                  S_ave_pos,
                                  tree_emb_dim=40,
                                  regularization_D=True):
        pyro.module("encoder", self.encoder)

        blosum = pyro.param("Blosum", self.B[:-1, :-1])

        alpha_loc = pyro.param("alpha_loc", torch.ones(3, device=self.device),
                               constraint=constraints.interval(0.1, 3))
        sigma_f_loc = pyro.param("sigma_f_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                                 constraint=constraints.interval(0.1, 3))
        sigma_n_loc = pyro.param("sigma_n_loc", torch.ones(self.n_z, device=self.device) * 0.2,
                                 constraint=constraints.interval(0.1, 3))
        lambd_loc = pyro.param("lambd_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                               constraint=constraints.interval(0.1, 3))

        tree_emb_len = pyro.param("tree_embed_len",
                                  torch.ones(Unaligned_S.shape[0], 1, device=self.device) * 0.5,
                                  constraint=constraints.interval(1e-3, 1 - 1e-3))
        with pyro.plate("tree_num", Unaligned_S.shape[0], dim=-2):
            with pyro.plate("tree_dim", tree_emb_dim, dim=-1):
                pyro.sample("tree_emb_len_beta", dist.Delta(tree_emb_len))

        with pyro.plate("hyper_alpha", 3):
            pyro.sample("alpha", dist.Delta(alpha_loc))
        with pyro.plate("sigma_lambda", self.n_z):
            pyro.sample("sigma_f", dist.Delta(sigma_f_loc))
            pyro.sample("sigma_n", dist.Delta(sigma_n_loc))
            pyro.sample("lambd", dist.Delta(lambd_loc))

        S_onehot = torch.zeros_like(Unaligned_S[:, :, None]).expand([-1, -1, self.n_c]).clone().scatter_(-1,
                                                                                                         Unaligned_S[:,
                                                                                                         :, None], 1)
        S_onehot = S_onehot[:, :, :].float().to(self.device)
        ref_emb = msa_ref
        mask = (torch.arange(Unaligned_S.shape[1])[None, :] < torch.tensor(S_length)[:, None]).to(self.device)
        s_emb = (S_onehot * mask[:, :, None])[:, :, :-1]

        similar_tensor = ref_emb @ blosum @ s_emb.transpose(1, 2)

        path = snw(similar_tensor, list(S_length), self.nw_fn)

        with pyro.plate("sequences_path", S_onehot.shape[0]):
            pyro.sample("path", dist.Delta(path).to_event(2))

        S = path @ S_onehot  # N, L, l @ N, l, n_c
        assert torch.all(S[:, :, -1] == 0)
        S[:, :, -1] = S[:, :, -1] + 1 - path.sum(-1)  # fill the gaps with onehot-encoding of "-"

        S_ave = torch.sum(S.detach(), dim=0, keepdim=True)[:, :, :-1]
        S_ave_pos.append(S_ave)

        Z_loc, Z_scale = self.encoder(S, None, return_scale=True)
        with pyro.plate("sequences_obs", S.shape[0], dim=-2):
            with pyro.plate("teacher_length", self.n_l, dim=-1):
                pyro.sample("teacher_S", Delta(S).to_event(1))
            with pyro.plate("msa_length", self.n_l, dim=-1):
                S = S + 1e-5
                S = S / S.sum(-1, True)
                pyro.sample("S", Delta(S).to_event(1))

        with pyro.plate("dim", self.n_z, dim=-1):
            pyro.sample("latent", dist.Normal(Z_loc.T, Z_scale.T).to_event(1))

    def batched_model(self,
                      Unaligned_S,
                      S_length,
                      msa_ref,
                      batch_distance,
                      S_ave_pos,
                      num_batch,
                      Unaligned_S_back,
                      S_back_length
                      ):
        pyro.module("decoder", self.decoder)

        # sample parameters of OU hyper-priors
        with pyro.plate("hyper_alpha", 3):
            alpha = pyro.sample("alpha", dist.Gamma(torch.ones(1, device=self.device) * 2,
                                                    torch.ones(1, device=self.device)).expand([3]))
        # sample from OU hyper-priors
        with pyro.plate("sigma_lambda", self.n_z):
            sigma_f = pyro.sample("sigma_f", dist.HalfNormal(alpha[0]).expand([self.n_z]))
            sigma_n = pyro.sample("sigma_n", dist.HalfNormal(alpha[1]).expand([self.n_z]))
            lambd = pyro.sample("lambd", dist.HalfNormal(alpha[2]).expand([self.n_z]))

        assert batch_distance.shape[0] == Unaligned_S_back.shape[0] + Unaligned_S.shape[0]

        cov = batch_distance[np.newaxis, :, :].expand(
            [self.n_z, batch_distance.shape[0], batch_distance.shape[0]]) / lambd[:, np.newaxis, np.newaxis]
        cov = (torch.exp(-cov) * sigma_f[:, np.newaxis, np.newaxis] ** 2 +
               torch.eye(cov.shape[1], device=self.device)[np.newaxis, :, :] * sigma_n[:, np.newaxis, np.newaxis] ** 2)

        cov_back = cov[:, :Unaligned_S_back.shape[0], :Unaligned_S_back.shape[0]]

        # sample z backbone
        with pyro.plate("dim_back", self.n_z, dim=-1):
            Z_back = pyro.sample("latent_back", dist.MultivariateNormal(
                torch.zeros(self.n_z, Unaligned_S_back.shape[0], device=self.device), cov_back))
        Z_back = Z_back.T

        with pyro.plate("sequences_path_back", Unaligned_S_back.shape[0], dim=-1):
            path_back = pyro.sample("path_back", ImproperUniform(torch.zeros(1,
                                                                             self.n_l,
                                                                             Unaligned_S_back.shape[1],
                                                                             device=self.device)).to_event(2))
        mask_back = (torch.arange(Unaligned_S_back.shape[1])[None, :] < torch.tensor(S_back_length)[:, None]).to(
            self.device)
        with pyro.plate("sequences_obs_back", Unaligned_S_back.shape[0], dim=-2):
            with pyro.plate("teacher_length_back", self.n_l, dim=-1):
                teachor_S_back = pyro.sample("teacher_S_back",
                                             ImproperUniform(torch.ones(Unaligned_S_back.shape[0],
                                                                        self.n_l,
                                                                        self.n_c,
                                                                        device=self.device)).to_event(1))
            loc_back = self.decoder(teachor_S_back, Z_back, True)
            with pyro.plate("msa_length_back", self.n_l, dim=-1):
                A_back = pyro.sample("S_back",
                                     dist.RelaxedOneHotCategoricalStraightThrough(torch.ones(1, device=self.device),
                                                                                  logits=loc_back))
            Unaligned_A_back = path_back.permute((0, 2, 1)) @ loc_back
            with pyro.plate("seq_length_back", Unaligned_S_back.shape[1], dim=-1):
                pyro.sample("obs_back",
                            dist.Categorical(logits=Unaligned_A_back).mask(mask_back),
                            obs=Unaligned_S_back)

        cov_batch = cov[:, Unaligned_S_back.shape[0]:, Unaligned_S_back.shape[0]:]
        cov_batch_back = cov[:, Unaligned_S_back.shape[0]:, :Unaligned_S_back.shape[0]]
        cov_back_inv = torch.linalg.inv(cov_back)
        mu_batch = (cov_batch_back @ cov_back_inv @ Z_back.T[:, :, None]).squeeze(-1)
        # n_z, N_batch, N_back @ n_z, N_back, N_back @ n_z, N_back, 1
        cov_batch = cov_batch - cov_batch_back @ cov_back_inv @ cov_batch_back.transpose(1, 2)

        with pyro.poutine.scale(None, num_batch):
            with pyro.plate("dim_batch", self.n_z, dim=-1):
                Z_batch = pyro.sample("latent_batch", dist.MultivariateNormal(mu_batch, cov_batch))
            Z_batch = Z_batch.T
            with pyro.plate("sequences_path_batch", Unaligned_S.shape[0], dim=-1):
                path_batch = pyro.sample("path_batch", ImproperUniform(torch.zeros(1,
                                                                                   self.n_l,
                                                                                   Unaligned_S.shape[1],
                                                                                   device=self.device)).to_event(2))
            mask_batch = (torch.arange(Unaligned_S.shape[1])[None, :] < torch.tensor(S_length)[:, None]).to(self.device)
            with pyro.plate("sequences_obs_batch", Unaligned_S.shape[0], dim=-2):
                with pyro.plate("teacher_length_batch", self.n_l, dim=-1):
                    teachor_S_batch = pyro.sample("teacher_S_batch",
                                                  ImproperUniform(torch.ones(Unaligned_S.shape[0],
                                                                             self.n_l,
                                                                             self.n_c,
                                                                             device=self.device)).to_event(1))
                loc_batch = self.decoder(teachor_S_batch, Z_batch, True)
                with pyro.plate("msa_length_batch", self.n_l, dim=-1):
                    A_batch = pyro.sample("S_batch",
                                          dist.RelaxedOneHotCategoricalStraightThrough(
                                              torch.ones(1, device=self.device),
                                              logits=loc_batch))
                Unaligned_A_batch = path_batch.permute((0, 2, 1)) @ loc_batch
                with pyro.plate("seq_length_batch", Unaligned_S.shape[1], dim=-1):
                    pyro.sample("obs_batch",
                                dist.Categorical(logits=Unaligned_A_batch).mask(mask_batch),
                                obs=Unaligned_S)

    def batched_guide(self,
                      Unaligned_S,
                      S_length,
                      msa_ref,
                      batch_distance,
                      S_ave_pos,
                      num_batch,
                      Unaligned_S_back,
                      S_back_length
                      ):
        pyro.module("encoder", self.encoder)
        blosum = pyro.param("Blosum", self.B[:-1, :-1])

        alpha_loc = pyro.param("alpha_loc", torch.ones(3, device=self.device),
                               constraint=constraints.interval(0.1, 3))
        sigma_f_loc = pyro.param("sigma_f_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                                 constraint=constraints.interval(0.1, 3))
        sigma_n_loc = pyro.param("sigma_n_loc", torch.ones(self.n_z, device=self.device) * 0.2,
                                 constraint=constraints.interval(0.1, 3))
        lambd_loc = pyro.param("lambd_loc", torch.ones(self.n_z, device=self.device) * 2.5,
                               constraint=constraints.interval(0.1, 3))

        with pyro.plate("hyper_alpha", 3):
            pyro.sample("alpha", dist.Delta(alpha_loc))
        with pyro.plate("sigma_lambda", self.n_z):
            pyro.sample("sigma_f", dist.Delta(sigma_f_loc))
            pyro.sample("sigma_n", dist.Delta(sigma_n_loc))
            pyro.sample("lambd", dist.Delta(lambd_loc))

        # backbone sequences
        S_onehot_back = torch.zeros_like(
            Unaligned_S_back[:, :, None]
        ).expand([-1, -1, self.n_c]).clone().scatter_(-1, Unaligned_S_back[:, :, None], 1)
        S_onehot_back = S_onehot_back[:, :, :].float().to(self.device)

        # mask_back = (torch.arange(Unaligned_S_back.shape[1])[None, :] < torch.tensor(S_back_length)[:, None]).to(
        #     self.device)
        # s_emb_back = self.contextual_emb_layer((S_onehot_back * mask_back[:, :, None]).permute((0, 2, 1))).permute(
        #     (0, 2, 1))
        #
        # similar_tensor_back = ref_emb @ s_emb_back.transpose(1, 2)

        # ref_emb = self.contextual_emb_layer(msa_ref[None, :, :].permute((0, 2, 1))).permute((0, 2, 1))[0]
        ref_emb = msa_ref
        mask_back = (torch.arange(Unaligned_S_back.shape[1])[None, :] < torch.tensor(S_back_length)[:, None]).to(self.device)
        # s_emb = self.contextual_emb_layer((S_onehot * mask[:, :, None])[:, :, :-1].permute((0, 2, 1))).permute((0, 2, 1))
        s_emb_back = (S_onehot_back * mask_back[:, :, None])[:, :, :-1]

        similar_tensor_back = ref_emb @ blosum @ s_emb_back.transpose(1, 2)


        path_back = snw(similar_tensor_back, list(S_back_length), self.nw_fn)
        with pyro.plate("sequences_path_back", S_onehot_back.shape[0]):
            pyro.sample("path_back", dist.Delta(path_back).to_event(2))

        S_back = path_back @ S_onehot_back  # N, L, l @ N, l, n_c
        assert torch.all(S_back[:, :, -1] == 0)
        S_back[:, :, -1] = S_back[:, :, -1] + 1 - path_back.sum(-1)  # fill the gaps with onehot-encoding of "-"

        if len(S_ave_pos) == 0:
            S_ave = torch.sum(S_back.detach(), dim=0, keepdim=True)[:, :, :-1]
            S_ave_pos.append(S_ave)

        Z_loc_back, Z_scale_back = self.encoder(S_back, None, return_scale=True)
        with pyro.plate("sequences_obs_back", S_back.shape[0], dim=-2):
            with pyro.plate("teacher_length_back", self.n_l, dim=-1):
                pyro.sample("teacher_S_back", Delta(S_back).to_event(1))
            with pyro.plate("msa_length_back", self.n_l, dim=-1):
                S_back = S_back + 1e-5
                S_back = S_back / S_back.sum(-1, True)
                pyro.sample("S_back", Delta(S_back).to_event(1))

        with pyro.plate("dim_back", self.n_z, dim=-1):
            pyro.sample("latent_back", dist.Normal(Z_loc_back.T, Z_scale_back.T).to_event(1))

        with pyro.poutine.scale(None, num_batch):
            S_onehot_batch = torch.zeros_like(
                Unaligned_S[:, :, None]
            ).expand([-1, -1, self.n_c]).clone().scatter_(-1, Unaligned_S[:, :, None], 1)
            S_onehot_batch = S_onehot_batch[:, :, :].float().to(self.device)

            # mask_batch = (torch.arange(Unaligned_S.shape[1])[None, :] < torch.tensor(S_length)[:, None]).to(self.device)
            # s_emb_batch = self.contextual_emb_layer(
            #     (S_onehot_batch * mask_batch[:, :, None]).permute((0, 2, 1))).permute((0, 2, 1))
            # similar_tensor_batch = ref_emb @ s_emb_batch.transpose(1, 2)


            mask_batch = (torch.arange(Unaligned_S.shape[1])[None, :] < torch.tensor(S_length)[:, None]).to(self.device)
            # s_emb = self.contextual_emb_layer((S_onehot * mask[:, :, None])[:, :, :-1].permute((0, 2, 1))).permute((0, 2, 1))
            s_emb_batch = (S_onehot_batch * mask_batch[:, :, None])[:, :, :-1]

            similar_tensor_batch = ref_emb @ blosum @ s_emb_batch.transpose(1, 2)


            path_batch = snw(similar_tensor_batch, list(S_length), self.nw_fn)


            with pyro.plate("sequences_path_batch", S_onehot_batch.shape[0]):
                pyro.sample("path_batch", dist.Delta(path_batch).to_event(2))

            S_batch = path_batch @ S_onehot_batch  # N, L, l @ N, l, n_c
            assert torch.all(S_batch[:, :, -1] == 0)
            S_batch[:, :, -1] = S_batch[:, :, -1] + 1 - path_batch.sum(-1)  # fill the gaps with onehot-encoding of "-"

            S_ave = torch.sum(S_batch.detach(), dim=0, keepdim=True)[:, :, :-1]
            S_ave_pos.append(S_ave)

            Z_loc_batch, Z_scale_batch = self.encoder(S_batch, None, return_scale=True)
            with pyro.plate("sequences_obs_batch", S_batch.shape[0], dim=-2):
                with pyro.plate("teacher_length_batch", self.n_l, dim=-1):
                    pyro.sample("teacher_S_batch", Delta(S_batch).to_event(1))
                with pyro.plate("msa_length_batch", self.n_l, dim=-1):
                    S_batch = S_batch + 1e-5
                    S_batch = S_batch / S_batch.sum(-1, True)
                    pyro.sample("S_batch", Delta(S_batch).to_event(1))

            with pyro.plate("dim_batch", self.n_z, dim=-1):
                pyro.sample("latent_batch", dist.Normal(Z_loc_batch.T, Z_scale_batch.T).to_event(1))

    def batched_model_latent_tree(self,
                                  Unaligned_S,
                                  S_length,
                                  S_index,
                                  msa_ref,
                                  S_ave_pos,
                                  num_batch,
                                  Unaligned_S_back,
                                  S_back_length,
                                  S_back_index,
                                  num_instance,
                                  tree_emb_dim=40,
                                  regularization_D=False):
        self.tree_emb_dim = tree_emb_dim
        tree_emb_loc = pyro.param("tree_embed_loc",
                                  torch.randn(num_instance, tree_emb_dim, device=self.device))

        tree_emb_loc_back = tree_emb_loc[S_back_index]
        with pyro.plate("tree_num_back", Unaligned_S_back.shape[0], dim=-2):
            with pyro.plate("tree_dim_back", tree_emb_dim, dim=-1):
                tree_emb_len = pyro.sample("tree_emb_len_beta_back", dist.Beta(torch.tensor([5.55], device=self.device),
                                                                               torch.tensor([2.28], device=self.device))
                                           )
        tree_embedding_back = tree_emb_loc_back / tree_emb_loc_back.norm(dim=-1, keepdim=True) * tree_emb_len

        # if regularization_D:
        #     gamma_beta = torch.tensor(1.5, device=self.device)
        #     gamma_alpha = torch.tensor(3.0, device=self.device)
        #     with pyro.plate("D_values_back", (batch_distance_back.shape[0] - 1) * batch_distance_back.shape[0] / 2):
        #         pyro.sample("D_back",
        #                     ImproperGamma(concentration=gamma_alpha, rate=gamma_beta),
        #                     obs=batch_distance_back[
        #                         torch.triu_indices(batch_distance_back.shape[0], batch_distance_back.shape[0], 1)[0],
        #                         torch.triu_indices(batch_distance_back.shape[0], batch_distance_back.shape[0], 1)[1]])

        with pyro.poutine.scale(None, num_batch):
            tree_emb_loc_batch = tree_emb_loc[S_index]
            with pyro.plate("tree_num_batch", Unaligned_S.shape[0], dim=-2):
                with pyro.plate("tree_dim_batch", tree_emb_dim, dim=-1):
                    tree_emb_len = pyro.sample("tree_emb_len_beta_batch",
                                               dist.Beta(torch.tensor([5.55], device=self.device),
                                                         torch.tensor([2.28], device=self.device))
                                               )
            tree_embedding_batch = tree_emb_loc_batch / tree_emb_loc_batch.norm(dim=-1, keepdim=True) * tree_emb_len
        tree_embedding = torch.cat([tree_embedding_back, tree_embedding_batch], dim=0)
        batch_distance = self._get_D(tree_embedding, 1e-5)
        self.batched_model(Unaligned_S,
                           S_length,
                           msa_ref,
                           batch_distance,
                           S_ave_pos,
                           num_batch,
                           Unaligned_S_back,
                           S_back_length
                           )

        #
        # if regularization_D:
        #     gamma_beta = torch.tensor(1.5, device=self.device)
        #     gamma_alpha = torch.tensor(3.0, device=self.device)
        #     with pyro.plate("D_values_batch", (batch_distance_batch.shape[0] - 1) * batch_distance_batch.shape[0] / 2):
        #         pyro.sample("D_batch",
        #                     ImproperGamma(concentration=gamma_alpha, rate=gamma_beta),
        #                     obs=batch_distance_batch[
        #                         torch.triu_indices(batch_distance_batch.shape[0], batch_distance_batch.shape[0], 1)[0],
        #                         torch.triu_indices(batch_distance_batch.shape[0], batch_distance_batch.shape[0], 1)[1]])

    def batched_guide_latent_tree(self,
                                  Unaligned_S,
                                  S_length,
                                  S_index,
                                  msa_ref,
                                  S_ave_pos,
                                  num_batch,
                                  Unaligned_S_back,
                                  S_back_length,
                                  S_back_index,
                                  num_instance,
                                  tree_emb_dim=40,
                                  regularization_D=True):
        tree_emb_len = pyro.param("tree_embed_len",
                                  torch.ones(num_instance, 1, device=self.device) * 0.5,
                                  constraint=constraints.interval(1e-3, 1 - 1e-3))

        tree_emb_len_back = tree_emb_len[S_back_index]
        with pyro.plate("tree_num_back", Unaligned_S_back.shape[0], dim=-2):
            with pyro.plate("tree_dim_back", tree_emb_dim, dim=-1):
                pyro.sample("tree_emb_len_beta_back", dist.Delta(tree_emb_len_back))

        with pyro.poutine.scale(None, num_batch):
            tree_emb_len_batch = tree_emb_len[S_index]
            with pyro.plate("tree_num_batch", Unaligned_S.shape[0], dim=-2):
                with pyro.plate("tree_dim_batch", tree_emb_dim, dim=-1):
                    pyro.sample("tree_emb_len_beta_batch", dist.Delta(tree_emb_len_batch))

        self.batched_guide(Unaligned_S,
                           S_length,
                           msa_ref,
                           None,
                           S_ave_pos,
                           num_batch,
                           Unaligned_S_back,
                           S_back_length
                           )

    def _get_D(self, tree_embedding, add_noise):
        batch_size = tree_embedding.shape[0]
        dp = (1 - torch.norm(tree_embedding, 2, dim=-1) ** 2)
        dn = dp.reshape(-1, 1) * dp.reshape(1, -1)
        d = 1 + 2 * torch.sum(
            (tree_embedding[:, np.newaxis, :].expand([batch_size, batch_size, self.tree_emb_dim]) -
             tree_embedding[np.newaxis, :, :].expand([batch_size, batch_size, self.tree_emb_dim])) ** 2,
            dim=-1) / dn
        batch_distance = torch.arccosh(d + add_noise)

        return batch_distance
