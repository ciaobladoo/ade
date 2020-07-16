import torch
import torch.nn as nn
from torch.nn import init

from modules import MLP


class MLPPotential(nn.Module):
    def __init__(self, dim, hidden_dims, act=nn.ReLU):
        super(MLPPotential, self).__init__()

        self.mlp = MLP(dim, 1, hidden_dims, non_linear_layer=act)

        for m in self.modules():
            if (
                    isinstance(m, nn.Linear)
                    or isinstance(m, nn.Conv2d)
                    or isinstance(m, nn.Conv1d)
            ):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class ExchangeablePotential(nn.Module):
    #TODO Initialization
    def __init__(self, dim, mlp_hidden_dims, d_model, nhead, dim_feedforward=2048, layer_norm=False,
                 num_layers=1, dropout=0.0, activation="relu", pool=torch.mean):
        super(ExchangeablePotential, self).__init__()

        self.mlp = MLP(dim, d_model, mlp_hidden_dims)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        if layer_norm:
            encoder_norm = nn.LayerNorm(d_model)
        else:
            encoder_norm = None
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.pool = pool
        self.score = MLP(d_model, 1, [])

    def forward(self, x):
        x = self.mlp(x)
        x = x.permute((1, 0, 2))
        return self.score(self.pool(self.encoder(x), dim=0)).squeeze(-1)
