from torch import nn
import torch
import numpy as np


class Encoder_ARVAE(nn.Module):
    def __init__(self, n_l, n_z=50, n_c=21):
        super().__init__()
        self.n_l = int(np.ceil(n_l/8) * 8)
        self.n_z = n_z
        self.n_c = n_c
        self.real_n_l = n_l
        modules = []
        for i in range(5):
            modules.append(nn.Conv1d(in_channels=self.n_c if i == 0 else self.n_c * (2 ** (i - 1)),
                                     out_channels=self.n_c * (2 ** i),
                                     kernel_size=2,
                                     stride=1 if i == 0 else 2,
                                     bias=False,    # because BN is used afterwards
                                     ))
            modules.append(nn.BatchNorm1d(num_features=self.n_c * (2 ** i)))
            modules.append(nn.PReLU())
        self.nets = nn.Sequential(*modules)
        self.dense_loc = nn.Linear((self.n_l - 1) // 16 * self.n_c * (2 ** 4), self.n_z)
        self.dense_scale = nn.Linear((self.n_l - 1) // 16 * self.n_c * (2 ** 4), self.n_z)

    def forward(self, X, X_length, return_scale=False):
        if self.n_l != self.real_n_l:
            padding = torch.zeros(X.shape[0], self.n_l - self.real_n_l, self.n_c, device=X.device)
            X = torch.cat([X, padding], dim=1)
        X = torch.permute(X, (0, 2, 1))
        assert X.shape[1:] == (self.n_c, self.n_l)   # check if the shape is (N, C, L)
        h = self.nets(X)
        h = nn.Flatten()(h)
        loc = self.dense_loc(h)
        scale = 1e-3 * nn.Softplus()(self.dense_scale(h))
        if return_scale:
            return loc, scale
        else:
            return loc


class Upsampler(nn.Module):
    def __init__(self, n_l, n_z=50, n_c=21, min_deconv_dim=42):
        super(Upsampler, self).__init__()
        self.n_l = int(np.ceil(n_l/8) * 8)
        self.n_z = n_z
        self.n_c = n_c
        self.real_n_l = n_l
        self.min_deconv_dim = min_deconv_dim
        self.low_res_features = min(min_deconv_dim * (2**3), 336)
        self.dense = nn.Linear(self.n_z, int(self.n_l // 8 * self.low_res_features))
        modules = []
        for i in range(3):
            in_channels = min(min_deconv_dim * 2 ** (3 - i), 336)
            out_channels = min(min_deconv_dim * 2 ** (3 - (i + 1)), 336)
            modules.append(nn.ConvTranspose1d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=2,
                                              stride=2,
                                              bias=False,    # because BN is used afterwards
                                              ))
            modules.append(nn.BatchNorm1d(num_features=out_channels))
            modules.append(nn.PReLU())
        self.nets = nn.Sequential(*modules)

    def forward(self, z):
        h = self.dense(z).reshape(-1, self.low_res_features, self.n_l // 8)
        h = self.nets(h)
        return h


class Decoder_ARVAE(nn.Module):
    def __init__(self, n_l, n_z=50, n_c=21, min_deconv_dim=42, dropout=0.45, gru_hidden_size=512, gru_num_layer=1):
        super().__init__()
        self.n_l = int(np.ceil(n_l/8) * 8)
        self.n_z = n_z
        self.n_c = n_c
        self.real_n_l = n_l
        self.gru_num_layer = gru_num_layer
        self.gru_hidden_size = gru_hidden_size
        self.dropout = nn.Dropout(dropout)
        self.project_x = nn.Conv1d(in_channels=n_c,
                                   out_channels=n_c,
                                   stride=1,
                                   kernel_size=1)
        self.upsampler = Upsampler(n_l, n_z, n_c, min_deconv_dim)
        self.rnn = nn.GRU(input_size=min(min_deconv_dim, 336) + n_c,
                          hidden_size=gru_hidden_size,
                          num_layers=self.gru_num_layer,
                          batch_first=True,
                          bidirectional=False)
        self.output_x = nn.Conv1d(in_channels=gru_hidden_size,
                                  out_channels=n_c,
                                  stride=1,
                                  kernel_size=1)

    # def forward(self, X, z, is_training=True):
    #
    #     h = self.upsampler(z)
    #     assert h.shape[-1] == self.n_l
    #     rnn_input = h # N * C * L
    #     rnn_input = torch.permute(rnn_input, (0, 2, 1)) # N * L * C
    #     rnn_output = self.rnn(rnn_input)[0] # N * L * H
    #     rnn_output = torch.permute(rnn_output, (0, 2, 1)) # N * H * L
    #     x_logits = self.output_x(rnn_output) # N * n_c * L
    #
    #     return torch.permute(x_logits, (0, 2, 1))[:, :self.real_n_l, :]

    def forward(self, X, z, is_training=True):
        if is_training:
            if self.n_l != self.real_n_l:
                padding = torch.zeros(X.shape[0], self.n_l - self.real_n_l, self.n_c, device=X.device)
                X = torch.cat([X, padding], dim=1)
            X = torch.permute(X, (0, 2, 1))
            h = self.upsampler(z)
            assert h.shape[-1] == self.n_l
            assert X.shape[1:] == (self.n_c, self.n_l)
            X = torch.cat([torch.zeros_like(X[:, :, 0:1]), X[:, :, :-1]], dim=-1)
            dropout_mask = self.dropout(torch.ones_like(X)[:, 0:1, :])
            projected_x = self.project_x(X * dropout_mask)
            rnn_input = torch.cat([h, projected_x], dim=-2) # N * C * L
            rnn_input = torch.permute(rnn_input, (0, 2, 1)) # N * L * C
            rnn_output = self.rnn(rnn_input)[0] # N * L * H
            rnn_output = torch.permute(rnn_output, (0, 2, 1)) # N * H * L
            x_logits = self.output_x(rnn_output) # N * n_c * L
        else:
            with torch.no_grad():
                h = self.upsampler(z)
                predicted_x = []
                rnn_input_x = torch.zeros(z.shape[0], self.n_c, 1, device=z.device)
                rnn_h_last = torch.zeros(self.gru_num_layer, z.shape[0], self.gru_hidden_size, device=z.device)
                for i in range(self.n_l):
                    rnn_input = torch.cat([h[:, :, i:i+1], self.project_x(rnn_input_x)], dim=-2)
                    rnn_input = torch.permute(rnn_input, (0, 2, 1))
                    rnn_output, rnn_h_last = self.rnn(rnn_input, rnn_h_last)
                    rnn_output = torch.permute(rnn_output, (0, 2, 1))
                    x_logit = self.output_x(rnn_output)
                    predicted_x.append(x_logit)
                    rnn_input_x = torch.zeros_like(rnn_input_x).scatter_(-2, torch.argmax(x_logit, dim=-2, keepdim=True), 1.0)
                x_logits = torch.cat(predicted_x, dim=-1)

        return torch.permute(x_logits, (0, 2, 1))[:, :self.real_n_l, :]

