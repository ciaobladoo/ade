import itertools
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.distributions as dist
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import data
import potentials
import flows
import hmc
# import utils


# arguments
data_dim = 2
dataset_generator = data.Exchangeable2DGaussian
modes = np.array([[-5.0, -5.0], [-2.0, 4.0], [7.0, -4.0], [3.0, 5.0]])

Potential = potentials.ExchangeablePotential
pot_hidden_dim = [128]*3

Flow = flows.PlanarFlow
flow_layers = 10

Dynamics = hmc.HMCNet
t_length = 1
mcmc_steps = 5
step_size = 1.0

batch_size = 100
num_epochs = 10
iters_per_epoch = 100
lr = 1e-3
sampler_iter = 5


def main():
    # creat data loader
    dataset = dataset_generator(modes.shape[0], num_epochs*iters_per_epoch*batch_size, modes)
    train_loader = data.get_loader(dataset, batch_size)
    train_loader_iter = iter(train_loader)
    test_loader = data.get_loader(dataset, 1000)
    true_data = next(iter(test_loader))

    # build model
    potential = Potential(data_dim, pot_hidden_dim, 256, 8, 1024).cuda()
    # flow = flows.NormalizingFlow(data_dim, Flow, flow_layers).cuda()
    lift = dist.multivariate_normal.MultivariateNormal(torch.zeros(modes.size).cuda(), torch.eye(modes.size).cuda())
    dynamic = Dynamics(data_dim, lift, t_length, mcmc_steps, step_size=step_size).cuda()

    def generator(potential, flow, dynamic, x, log_px):
        # x, log_px = flow(x, log_px)
        xt, log_px = dynamic(x, potential, log_px)
        return xt, log_px

    sampler = lambda z, log_pz : generator(potential, None, dynamic, z, log_pz)

    # create optimizer
    optimizerP = optim.Adam(potential.parameters(), lr = lr)
    # optimizerG = optim.Adam(itertools.chain(*[flow.parameters(), dynamic.parameters()]), lr = lr)
    optimizerG = optim.Adam(itertools.chain(dynamic.parameters()), lr=lr)

    # loss functions
    def loss_G(z0, potential, flow, log_pz0):
        x_fake, log_px = flow(z0, log_pz0)
        fx_fake = -potential(x_fake)
        loss = (-fx_fake + log_px).mean()
        return loss

    def loss_P(x, z0, potential, flow):
        x_fake, log_px = flow(z0, 0)
        fx = -potential(x)
        fx_fake = -potential(x_fake.detach())
        # debug
        log_px = 0.0

        loss = (-fx + fx_fake - log_px).mean()
        return loss

    train_writer = SummaryWriter("")

    # training loop
    torch.backends.cudnn.benchmark = True

    init_dist = dist.multivariate_normal.MultivariateNormal(torch.zeros(modes.size).cuda(),
                                                            10.0*torch.eye(modes.size).cuda())

    for i in range(num_epochs):
        pbar = tqdm(range(iters_per_epoch), unit='batch')
        for _ in pbar:
            batch = next(train_loader_iter).float().cuda()

            # train potential
            optimizerP.zero_grad()
            z0 = init_dist.sample((batch.size(0),))
            z0.requires_grad = True
            z0 = z0.view((-1, *modes.shape))
            lossP = loss_P(batch, z0, potential, sampler)
            lossP.backward()
            optimizerP.step()

            # train sampler
            for _ in range(sampler_iter):
                optimizerG.zero_grad()
                z0 = init_dist.sample((batch.size(0),))
                z0.requires_grad = True
                log_prob_z0 = init_dist.log_prob(z0)
                z0 = z0.view((-1, *modes.shape))
                lossG = loss_G(z0, potential, sampler, log_prob_z0)
                lossG.backward()
                optimizerG.step()

            pbar.set_description('disc_loss: %.4f, gen_loss: %.4f' % (lossP.cpu().item(), lossG.cpu().item()))

        # save training statistics
        train_writer.add_scalar("metric/energy-loss", lossP.cpu().item(), global_step=i)
        train_writer.add_scalar("metric/generator-loss", lossG.cpu().item(), global_step=i)

        # mmd = utils.MMD(true_data, x, gamma)
        # print(mmd)
        # train_writer.add_scalar("eval/mmd", mmd, global_step=i)

    # evaluation
    z0 = init_dist.sample((1000,))
    z0 = z0.view((-1, *modes.shape))
    z0.requires_grad = True
    x, _ = sampler(z0, 0)
    x = x.cpu().detach().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for e in range(modes.shape[0]):
        ax1.scatter(true_data[:, e, 0], true_data[:, e, 1])
        ax2.scatter(x[:, e, 0], x[:, e, 1])
    plt.show()


if __name__ == '__main__':
    main()
