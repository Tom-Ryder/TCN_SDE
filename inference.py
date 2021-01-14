import torch
from torch import nn
import torch.distributions as d
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import random
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
from pathlib import Path
import shutil

torch.manual_seed(0)

my_parser = argparse.ArgumentParser(description='List options for the experiment')
my_parser.add_argument('EXP_NAME',
                       metavar='EXP_NAME',
                       type=str,
                       help='the tensorboard run name')
my_parser.add_argument('CUDA_ID',
                       metavar='CUDA_ID',
                       type=str,
                       help='the CUDA ID')
args = my_parser.parse_args()

STATE_DIM = 3
dt = .2
T = 10
N = int(T/dt)+1
BATCH_SIZE = 1

device = torch.device("".join(["cuda:",f'{args.CUDA_ID}']) if torch.cuda.is_available() else "cpu")
LR = 1e-3
niter = 1000000

obs_times = np.arange(0, T+dt, T)
obs_mean = torch.Tensor([[[-5.0, -5.0], [5.0, 5.0], [15., 15.]]]).to(device)
obs_std = torch.ones_like(obs_mean).to(device)

writer = SummaryWriter(f'runs/{args.EXP_NAME}')

def alpha(x, theta_x=(1.0, -5.0, 3.0), theta_y=(3.0, 5.0, 1.0), theta_z=(3.0, 15.0, 1.0)):
    x, y, z = torch.chunk(x, 3, 1)
    xdot = theta_x[0]*(theta_x[1] - x)
    ydot = theta_y[0]*(theta_y[1] - y)
    zdot = theta_z[0]*(theta_z[1] - z)
    return torch.cat([xdot, ydot, zdot], 1)

def beta_sqrt(x, theta_x=(1.0, -5.0, 3.0), theta_y=(3.0, 5.0, 1.0), theta_z=(3.0, 15.0, 1.0)):
    x, y, z = torch.chunk(torch.ones_like(x, device=x.device), 3, 1)
    xdot = theta_x[2]*x
    ydot = theta_y[2]*y
    zdot = theta_z[2]*z
    return torch.cat([xdot, ydot, zdot], 1)

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        b = b.type(inputs.dtype)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None

class MaskedConv1d(nn.Conv1d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv1d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kW // 2 + 1*(mask_type == 'B'):] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv1d, self).forward(x)

class ResNetBlock(nn.Module):

    def __init__(self, inp_cha, out_cha, stride = 1, batch_norm=True):
        super().__init__()
        self.conv1 = MaskedConv1d('B', inp_cha,  out_cha, 15, stride, 7, bias=False)
        self.conv2 = MaskedConv1d('B', out_cha,  out_cha, 15, 1, 7, bias=False)

        self.act1 = nn.PReLU(out_cha, init = 0.2)
        self.act2 = nn.PReLU(out_cha, init = 0.2)

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(out_cha)
            self.bn2 = nn.BatchNorm1d(out_cha)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        # If dimensions change, transform shortcut with a conv layer
        if inp_cha != out_cha or stride > 1:
            self.conv_skip = MaskedConv1d('B', inp_cha,  out_cha, 3, stride, 1, bias=False)
        else:
            self.conv_skip = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x) + self.conv_skip(residual)))
        return x

class ResNetBlockUnMasked(nn.Module):

    def __init__(self, inp_cha, out_cha, stride = 1, batch_norm=True):
        super().__init__()
        self.conv1 = nn.Conv1d(inp_cha,  out_cha, 15, stride, 7)
        self.conv2 = nn.Conv1d(out_cha,  out_cha, 15, 1, 7)

        self.act1 = nn.PReLU(out_cha, init = 0.2)
        self.act2 = nn.PReLU(out_cha, init = 0.2)

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(out_cha)
            self.bn2 = nn.BatchNorm1d(out_cha)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        # If dimensions change, transform shortcut with a conv layer
        if inp_cha != out_cha or stride > 1:
            self.conv_skip = nn.Conv1d(inp_cha,  out_cha, 15, stride, 7, bias=False)
        else:
            self.conv_skip = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x) + self.conv_skip(residual)))
        return x

class CouplingLayer(nn.Module):

    def __init__(self, cond_inputs, stride):
        super().__init__()
        self.net = nn.Sequential(ResNetBlock(1+cond_inputs, 96),
                                 MaskedConv1d('B', 96,  2, 15, stride, 7, bias=False))

        self.feature_net = nn.Sequential(ResNetBlockUnMasked(cond_inputs, 96),
                                          ResNetBlockUnMasked(96, cond_inputs))

        self.unpack = True if cond_inputs > 1 else False

    def forward(self, x, cond_inputs):
        if self.unpack:
            cond_inputs = torch.cat([*cond_inputs], 1)
        cond_inputs = self.feature_net(cond_inputs)
        feature_vec = torch.cat([x, cond_inputs], 1)
        output = self.net(feature_vec)
        mu, sigma = torch.chunk(output, 2, 1)
        mu = self._pass_through_units(mu)
        sigma = self._pass_through_units(sigma, mu=False)
        x = mu + sigma*x
        return x, sigma

    def _pass_through_units(self, params, mu=True):
        B, _, L = params.shape
        padding = 2 if STATE_DIM % 2==1 else 1
        if mu:
            pad = torch.zeros([B, padding, L], device=params.device)
        else:
            params = LowerBound.apply(params, 1e-6)
            pad = torch.ones([B, padding, L], device=params.device)
        return torch.cat([pad, params], 1).transpose(2, 1).reshape(B, 1, -1)

class PermutationLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.index = torch.randperm(STATE_DIM)

    def forward(self, x):
        B, S, L = x.shape
        x_reshape = x.reshape(B, S, -1, STATE_DIM)
        x_perm = x_reshape[:, :, :, self.index]
        x = x_perm.reshape(B, S, L)
        return x

class SDEFlow(nn.Module):

    def __init__(self, cond_inputs=1):
        super().__init__()

        stride = 3 if STATE_DIM % 2==1 else 2

        self.CL_1 = CouplingLayer(cond_inputs, stride)
        self.CL_2 = CouplingLayer(cond_inputs, stride)
        self.CL_3 = CouplingLayer(cond_inputs, stride)
        self.CL_4 = CouplingLayer(cond_inputs, stride)
        self.CL_5 = CouplingLayer(cond_inputs, stride)

        self.P_1 = PermutationLayer()
        self.P_2 = PermutationLayer()
        self.P_3 = PermutationLayer()
        self.P_4 = PermutationLayer()

        self.base_dist = d.normal.Normal(loc = 0., scale = 1.0)

    def forward(self, batch_size, *args, **kwargs):

        eps = self.base_dist.sample([batch_size, 1, STATE_DIM*N]).to(device)
        log_prob = self.base_dist.log_prob(eps).sum(-1)

        times = torch.arange(0, T+dt, dt, device=eps.device)[(None,)*2].repeat(batch_size, STATE_DIM, 1).transpose(-2, -1).reshape(batch_size, 1, -1)

        CL_1, CL_1_sigma = self.CL_1(eps, times)
        P_1 = self.P_1(CL_1)

        CL_2, CL_2_sigma = self.CL_2(P_1, times)
        P_2 = self.P_2(CL_2)

        CL_3, CL_3_sigma = self.CL_3(P_2, times)
        P_3 = self.P_3(CL_3)

        CL_4, CL_4_sigma = self.CL_4(P_3, times)
        P_4 = self.P_4(CL_4)

        y, CL_5_sigma = self.CL_5(P_4, times)

        for sigma in [CL_1_sigma, CL_2_sigma, CL_3_sigma, CL_4_sigma, CL_5_sigma]:
            log_prob -= torch.log(sigma).sum(-1)

        return y.reshape(batch_size, STATE_DIM, -1), log_prob

class ObsModel(nn.Module):

    def __init__(self, times, mu, scale):
        super().__init__()

        self.idx = self._get_idx(times)
        self.times = times
        self.mu = mu
        self.scale = scale
        self.obs_to_pad = self._obs_inputs()

    def forward(self, x):
        obs_ll = d.normal.Normal(self.mu, self.scale).log_prob(x[:, :, self.idx])
        return torch.sum(obs_ll, [-1, -2]).mean()

    def _get_idx(self, times):
        return list((times/dt).astype(int))

    def plt_dat(self):
        return self.mu, self.times

    def _obs_inputs(self):
        out = torch.zeros([1, STATE_DIM, N], device=self.mu.device)
        out[:, :, self.idx] = self.mu
        return out

    def get_obs_fts(self, batch_size):
        return self.obs_to_pad.repeat(batch_size, 1, 1).transpose(-2, -1).reshape(batch_size, 1, -1)

def calc_negative_elbo(log_prob, x, alpha, beta_sqrt):
    euler_maruyama = d.normal.Normal(loc=x[:, :, :-1] + alpha(x[:, :, :-1])*dt, scale = beta_sqrt(x[:, :, :-1])*math.sqrt(dt))
    return (log_prob - torch.sum(euler_maruyama.log_prob(x[:, :, 1:]).sum(-1), -1, keepdim=True)).mean()

class PosteriorPlotter():
    def __init__(self, run_every=250):
        self.run_every = run_every

    def plot(self, i, model, obs, n=30, path_dir=f"tmp_out.png"):
        if i % self.run_every == 0:
            with torch.no_grad():
                x, _ = model(n)
            obs_mu, obs_times = obs.plt_dat()
            fig, axs = plt.subplots(STATE_DIM, 1)
            obs_mu = obs_mu.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            t = np.arange(0, T+dt, dt)
            for j in range(x.shape[0]):
                for k in range(x.shape[1]):
                    axs[k].plot(t, x[j, k, :], color="black", alpha=0.5)
                    if j == 0:
                        axs[k].scatter(obs_times, obs_mu[0, k, :], color="red", marker="x")
            fig.savefig(path_dir, dpi=500)
            plt.close("all")

class TrainLogger():
    def __init__(self, run_every=50):
        self.run_every = run_every

    def log(self, i, metrics):
        if i % self.run_every == 0:
            for key, value in metrics.items():
                writer.add_scalar(f'train/{key}', value, i)

def train():
    net = SDEFlow().to(device)
    obs = ObsModel(times=np.arange(0, T+dt, T), mu=obs_mean, scale=obs_std).to(device)

    optimizer = optim.Adam(net.parameters(), lr=LR)

    train_logger = TrainLogger(run_every=50)
    posterior_plot = PosteriorPlotter(run_every=250)

    with tqdm(total=niter, desc=f'Train Diffusion', position=-1) as t:
            for train_iter in range(niter):
                net.train()
                optimizer.zero_grad()

                x, log_prob = net(BATCH_SIZE)
                obs_loss = obs(x)
                ELBO = calc_negative_elbo(log_prob, x, alpha, beta_sqrt)

                loss = ELBO - obs_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.50)
                optimizer.step()

                metrics = {'Negative ELBO': loss, 'Observation Log-Lik': obs_loss, 'Diffusion Loss': ELBO}

                train_logger.log(train_iter, metrics)
                posterior_plot.plot(train_iter, net, obs)
                t.update()

if __name__=="__main__":
    train()
