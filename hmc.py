import torch
import torch.nn as nn


def clip_vec(v, max_norm = 1.0, axis = -1):
    l2_norm = torch.norm(v, p=None, dim=axis, keepdim=True).detach()
    l2_norm = torch.clamp(l2_norm, min=max_norm)
    v = v*max_norm/l2_norm
    return v


class LeapFrog(nn.Module):
    def __init__(self):
        super(LeapFrog, self).__init__()

    def forward(self, x, v, potential, u, step_size):
        assert x.shape == v.shape
        dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v = v - 0.5*step_size*clip_vec(dx)
        x = x + step_size*clip_vec(v)
        u = potential(x)
        dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v = v - 0.5*step_size*clip_vec(dx)
        return x, v, u


class HMCNet(nn.Module):
    """
    HAMILTONIAN Markov Transitions
    """
    def __init__(self, dim, lift, t_length, mcmc_steps, integrator='leapfrog', step_size_mode = 'learn', step_size = 0.1, t_sampling='final', correction='mh'):
        """
        :param potential (torch.nn.Module): potential function implement as nn.Module
        :param lift (torch.distribution): conditional distribution for the momentum
        :param integrator (string): choice of numerical integrator for Markov transition
        """
        super(HMCNet, self).__init__()

        self.lift = lift
        self.t_sampling = t_sampling
        self.t_length = t_length
        self.correction = correction
        self.mcmc_steps = mcmc_steps

        # construct integrator
        if integrator == 'leapfrog':
            self.integrator = LeapFrog()
        if step_size_mode == 'learn':
            self.step_size = torch.nn.Parameter(torch.empty((mcmc_steps, dim)).fill_(step_size))
        else:
            self.step_size = torch.empty((mcmc_steps, dim)).fill_(step_size)

    def forward(self, x, potential, log_px):
        #sample = []
        x_current = x
        for step in range(self.mcmc_steps):
            x = x_current
            # TODO optimizing covariance of Gaussian?
            v = self.lift.sample((x.size(0),))
            u = potential(x)
            # H_current = u - self.lift.log_prob(v)
            H_current = u + 0.5*(v**2).sum(-1)

            # sampling trajectory and propose state
            # TODO clip sample and gradient
            # TODO adaptive step size adjustment (learning step size); adaptive T adjustment
            # TODO sample scheme other than 'final'
            v = v.view(x.shape)
            if self.t_sampling == 'final':
                for _ in range(self.t_length):
                    x, v, u = self.integrator(x, v, potential, u, self.step_size[step])
            # TODO matrix data
            # c_norm = 0.5*(v**2).sum(-1)
            # H_propose = u - self.lift.log_prob(-v)
            v = v.view(v.size(0), -1)
            H_propose = u + 0.5 * (v ** 2).sum(-1)

            # correction
            if self.correction == 'mh':
                ratio = torch.exp(H_current - H_propose)
                is_accept = torch.rand(1, device=ratio.device) < ratio
                is_accept_expand = is_accept[:,None,None].expand_as(x)
                x_current = torch.where(is_accept_expand, x, x_current)
                # c_norm *= is_accept
                # if is_accept:
                #    sample.append(x_current)
            else:
                x_current = x
                # sample.append(x_current)
            # log_px += c_norm

        return x_current, log_px
