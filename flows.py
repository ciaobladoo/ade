import math

import torch
import torch.nn as nn


class NormalizingFlow(nn.Module):
    def __init__(self, dim, flow, num_layers):
        super(NormalizingFlow, self).__init__()

        flows = []
        for i in range(num_layers):
            flows.append(flow(dim))
        self.flows = nn.Sequential(*flows)

    def forward(self, x, log_px):
        for flow in self.flows:
            x, log_px = flow(x, log_px)
        return x, log_px


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()

        sdv = 1. / math.sqrt(dim)
        self.u = nn.Parameter(torch.zeros(dim, 1).uniform_(-sdv, sdv))
        self.w = nn.Parameter(torch.zeros(dim, 1).uniform_(-sdv, sdv))
        self.b = nn.Parameter(torch.zeros(1, ))
        self.h = nn.Tanh()

    def forward(self, x, log_px):
        a = self.h(torch.matmul(x, self.w) + self.b)
        psi = torch.matmul(1-a**2, torch.t(self.w))

        v = torch.matmul(torch.t(self.w), self.u)
        m = torch.nn.Softplus()(v) - 1.0
        u = self.u + (m-v)*self.w / torch.matmul(torch.t(self.w), self.w)

        log_px = log_px - torch.log(1 + torch.matmul(psi,u)).squeeze(-1)
        x = x + torch.matmul(a, torch.t(u))

        return x, log_px
