import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    For Encoder and Decoders with an MLP architecture
    """
    def __init__(self, in_size, out_size, hidden_sizes, use_bias = True, use_non_linear_output=False, out_shape=None, non_linear_layer=nn.ReLU):
        super().__init__()
        if out_shape:
            assert np.prod(np.array(out_shape)) == out_size
            self.out_shape = out_shape
        else:
            self.out_shape = (out_size, )
        layers = []
        in_sizes = [in_size] + hidden_sizes
        out_sizes = hidden_sizes + [out_size]
        sizes = list(zip(in_sizes, out_sizes))
        for (i, o) in sizes[0:-1]:
            layers.append(nn.Linear(i, o, bias=use_bias))
            layers.append(non_linear_layer())
        layers.append(nn.Linear(sizes[-1][0], sizes[-1][1], bias=use_bias))
        if use_non_linear_output:
            layers.append(non_linear_layer())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        output_shape = x.size()[:-1] + self.out_shape
        return self.seq(x).view(output_shape)


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention Module
    """
    def __init__(self, input_shape, latent_shape, head_count, head_shape, output_shape):
        super().__init__()
        self.head_count = head_count
        self.latent_shape = latent_shape
        self.cts_encoder = MLP(input_shape, latent_shape*head_count, [], use_bias=False)
        self.ngh_encoder = MLP(input_shape, latent_shape*head_count, [], use_bias=False)
        self.com_encoder = MLP(input_shape, head_shape, [], use_bias=False)
        self.grp_encoder = MLP(head_count*head_shape, output_shape, [], use_bias=False)
        self.mlp = MLP(output_shape+input_shape, input_shape, [256, 256, 256, 256])

    def forward(self, x):
        (batch_size, n_roi) = x.size()[:2]
        cts = self.cts_encoder(x).view(batch_size, -1, self.head_count, self.latent_shape).permute((0,2,1,3))\
            .contiguous().view(-1, n_roi, self.latent_shape)
        ngh = self.ngh_encoder(x).view(batch_size, -1, self.head_count, self.latent_shape).permute((0,2,1,3))\
            .contiguous().view(-1, n_roi, self.latent_shape)
        weight = torch.bmm(cts, ngh.permute((0,2,1))).view(batch_size, -1, n_roi)
        weight = nn.Softmax(-2)(weight)
        # x = self.com_encoder(x).view(batch_size, n_roi, self.head_count,-1).permute((0,2,1,3))
        y = self.com_encoder(x).unsqueeze(-2).expand(batch_size, n_roi, self.head_count, -1)
        y = torch.matmul(weight.view(batch_size, self.head_count, n_roi, -1).permute((0,1,3,2)), y.permute((0,2,1,3)))
        y = self.grp_encoder(y.permute((0,2,1,3)).contiguous().view(batch_size,n_roi,-1))
        x = torch.cat((x,y), -1)
        x = self.mlp(x)

        return x
