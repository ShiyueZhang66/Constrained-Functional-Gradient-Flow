import argparse
import logging
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch import nn
import matplotlib.pyplot as plt
import pdb
import math
import tensorflow_probability as tfp
from scipy.spatial.distance import pdist, squareform
from torch.distributions.gamma import Gamma
from torch.utils.data import WeightedRandomSampler,DataLoader
import sklearn
import torch.distributions as Distributions

seed = 1224
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#real diabetes
from sklearn.datasets import load_diabetes

data_diabetes = load_diabetes()

diabetes_X = data_diabetes['data']
diabetes_X=sklearn.preprocessing.scale(diabetes_X, axis=0, with_mean=True, with_std=True, copy=True)
diabetes_X=torch.from_numpy(diabetes_X).float()

diabetes_y= data_diabetes['target']
diabetes_y=sklearn.preprocessing.scale(diabetes_y, axis=0, with_mean=True, with_std=False, copy=True)
diabetes_y=torch.from_numpy(diabetes_y).float()

diabetes_y=torch.unsqueeze(diabetes_y,dim=1)

N=diabetes_X.shape[0]
D=diabetes_X.shape[1]

ols=torch.mm(torch.mm(torch.inverse(torch.mm(diabetes_X.T,diabetes_X)),diabetes_X.T),diabetes_y)
print(ols)
print(ols.norm(p=1))

beta_optimal = torch.mm(torch.mm(torch.inverse(torch.mm(diabetes_X.T, diabetes_X) + torch.eye(D)), diabetes_X.T),
                        diabetes_y)

diabetes_y=torch.squeeze(diabetes_y)



def svgd_kernel(theta, h=-1):
    theta=theta.detach().numpy()
    sq_dist = pdist(theta)
    pairwise_dists = squareform(sq_dist) ** 2
    if h < 0:  # if h < 0, using median trick
        h = np.median(pairwise_dists)
        h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

    # compute the rbf kernel
    Kxy = np.exp(-pairwise_dists / h ** 2 / 2)

    dxkxy = -np.matmul(Kxy, theta)
    sumkxy = np.sum(Kxy, axis=1)
    for i in range(theta.shape[1]):
        dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
    dxkxy = dxkxy / (h ** 2)

    Kxy = torch.tensor(Kxy,dtype=torch.float32)
    dxkxy=torch.tensor(dxkxy,dtype=torch.float32)
    return (Kxy, dxkxy)


latent_dim_n=50

class F_net(nn.Module):
    def __init__(self, z_dim, latent_dim = latent_dim_n):
        super().__init__()
        self.z_dim = z_dim
        self.latent_dim = latent_dim

        self.dnn = nn.Sequential(nn.Linear(self.z_dim, self.latent_dim),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(self.latent_dim, self.latent_dim),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(self.latent_dim, self.z_dim))
    def forward(self, z):
        z = self.dnn(z)
        return z

class G_net(nn.Module):
    def __init__(self, z_dim, latent_dim = latent_dim_n):
        super().__init__()
        self.z_dim = z_dim
        self.latent_dim = latent_dim

        self.dnn = nn.Sequential(nn.Linear(self.z_dim, self.latent_dim),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(self.latent_dim, self.latent_dim),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(self.latent_dim, 1))
    def forward(self, z):
        z = self.dnn(z)
        return z


class ConstrainedGWG(object):
    def __init__(self, args=None):
        self.args = args
        self.device = "cpu"
        self.f_net = F_net(D,f_latent_dim).to(self.device)
        self.g_net = G_net(D,f_latent_dim).to(self.device)
        self.z_net = G_net(D,f_latent_dim).to(self.device)
        self.edge_width = edge_width
    def divergence_approx(self, fnet_value, parti_input, e=None):
        def sample_rademacher_like(y):
            return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1
        if e is None:
            e = sample_rademacher_like(parti_input)
        e_dzdx = torch.autograd.grad(fnet_value, parti_input, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = e_dzdx_e.view(parti_input.shape[0], -1).sum(dim=1)
        return approx_tr_dzdx


    def potential(self, theta,ts,sigma2,y=diabetes_y, X=diabetes_X):
        K = theta.size(0)
        beta = torch.abs(theta) * theta
        beta = beta[0:K-1] * ts
        loglik = -torch.norm(y.unsqueeze(1) - torch.mm(X,beta))**2/ (2*sigma2)
        logpri = -torch.norm(beta)**2 / (2*sigma2)
        U= -(loglik + logpri)
        return U


    def d_potential(self, theta,ts,sigma2,y=diabetes_y, X=diabetes_X):
        K = theta.size(0)
        beta = torch.abs(theta) * theta
        beta = beta[0:K-1] * ts
        dloglik = torch.mm(X.T, (y.unsqueeze(1) - torch.mm(X, beta))) / sigma2
        dlogpri = -beta / sigma2
        dU=-(dloglik + dlogpri)* ts * 2 * torch.abs(theta[0:K-1])
        dUt = torch.cat([dU, torch.tensor(0).unsqueeze(dim=0).unsqueeze(dim=1)]) - theta * torch.mm(theta[0:K-1].T, dU)
        return dUt

    def sph_hmc(self,current_q,ts,sigma2,eps=0.2, L=5):
        length = current_q.size(0)
        current_q = torch.reshape(current_q, (length, 1))

        q = current_q
        tst=ts

        # sample velocity
        v = torch.randn((length,1))  # standard multinormal

        v = v - q * (torch.mm(q.T, v))  # force v to lie in tangent space of sphere

        # Evaluate potential and kinetic energies at start of trajectory
        current_E = self.potential(q,tst,sigma2) + 0.5 * torch.norm(v) ** 2
        randL = torch.ceil(L * torch.rand(1))
        randL=int(randL)

        # Alternate full steps for position and momentum
        for i in range(randL):
            # Make a half step for velocity
            v = v - eps / 2 * self.d_potential(q,tst,sigma2)

            # Make a full step for the position
            q_0 = q
            v_nom = torch.norm(v)
            q = q_0 * torch.cos(v_nom * eps) + v / v_nom * torch.sin(v_nom * eps)
            v = -q_0 * v_nom * torch.sin(v_nom * eps) + v * torch.cos(v_nom * eps)

            # Make last half step for velocity
            v = v - eps / 2 * self.d_potential(q,tst,sigma2)

        # Evaluate potential and kinetic energies at end of trajectory
        proposed_E = self.potential(q,tst,sigma2) + 0.5 * torch.norm(v) ** 2

        # Accept or reject the state at end of trajectory, returning either
        # the position at the end of the trajectory or the initial position
        logAP = -proposed_E + current_E

        if logAP > min(0, torch.log(torch.rand(1))):
            Ind = 1
            return (q, Ind)
        else:
            Ind = 0
            return (current_q, Ind)



    def sph_train(self):
        NSamp = 11000
        NBurnIn = 1000
        Nsf = 10

        beta_plot = torch.zeros(10, D)

        t = torch.linspace(0.1, 1, steps=10)
        t = t * ols.norm(p=1)
        for sf in range(10):
            Samp = torch.zeros(NSamp-NBurnIn, D)
            TrjLength = torch.zeros(1000)
            Stepsz = torch.zeros(1000)
            wt = torch.zeros(NSamp-NBurnIn)
            start = time.time()

            ## HMC setting
            TrjLength[sf] = 2 * math.pi / D * (1-(sf+1) / Nsf / 4 * 3)
            NLeap = 10
            Stepsz[sf] = TrjLength[sf] / NLeap

            ## Initialization
            beta = torch.zeros(D) / t[sf]
            theta = torch.sign(beta) * torch.sqrt(torch.abs(beta))
            theta = torch.cat([theta, torch.sqrt(1-torch.norm(theta)**2).unsqueeze(dim=0)])
            beta = beta.unsqueeze(1)

            var = torch.norm(diabetes_y.unsqueeze(1) - torch.mm(diabetes_X, beta)) ** 2 + torch.norm(beta) ** 2
            sigma2 = 1 / torch.distributions.Gamma((N - 1 + D) / 2, var / 2).sample()

            for Iter in range(NSamp):
                # Use Spherical HMC to get sample theta
                samp,Ind = self.sph_hmc(theta,t[sf],sigma2, Stepsz[sf], NLeap)
                theta = samp
                # accp = accp + samp$Ind
                beta = torch.abs(theta) * theta
                beta = beta[0:D] * t[sf]

                # sample sigma2
                var = torch.norm(diabetes_y.unsqueeze(1) - torch.mm(diabetes_X, beta)) ** 2 + torch.norm(beta) ** 2
                sigma2 = 1 / torch.distributions.Gamma((N - 1 + D) / 2, var / 2).sample()

                # save sample beta
                if Iter+1 > NBurnIn:
                    beta=beta.squeeze(dim=1)
                    Samp[Iter - NBurnIn] = beta
                    wt[Iter - NBurnIn] = torch.log((2 * t[sf]))*D + torch.sum(torch.log(torch.abs(theta)))

            max_exp=torch.max(wt)
            wt=torch.exp(wt-max_exp)


            # Resample
            sampler = list(WeightedRandomSampler(wt, NSamp - NBurnIn,replacement=True))
            ReSamp = Samp[sampler] # Resample
            # ReSamp = Samp #non-resample
            # ReSamp=torch.tensor(ReSamp)
            ReSamp = ReSamp.clone().detach()

            beta_lasso = torch.median(ReSamp,0).values
            print(beta_lasso,'beta_lasso')

            beta_plot[sf] = beta_lasso

            error = torch.norm(beta_lasso - ols.squeeze(dim=1), p=1)
            print(error, 'l1 error')

            run_time = time.time() - start
            print(run_time, sf)


        # real
        zeros = torch.zeros(1, 10)
        beta_plot = torch.cat((zeros, beta_plot), dim=0)

        beta_plot_save = np.array(beta_plot)

        fig = plt.figure()
        ax = fig.add_subplot()
        for i in range(10):
            ax.plot(torch.linspace(0, 1, steps=11), beta_plot[:, i], label=i + 1,
                    linestyle='-')

        ax.legend()
        plt.legend(loc="upper right")
        plt.show()


    #cfg
    def score(self, beta, y=diabetes_y, X=diabetes_X, sigma2=100, t=0):

        beta_randint = torch.randint(0, num_particle, (1,))
        beta_pred = beta[:, beta_randint]

        # beta_pred = torch.median(beta, dim=1).values
        # beta_pred =beta_pred[:,None]

        var=torch.norm(y.unsqueeze(1) - torch.mm(X,beta_pred))**2+torch.norm(beta_pred)**2
        sigma2=1/torch.distributions.Gamma((N-1+D)/2,var/2).sample()

        dloglik = torch.mm(X.T,(y.unsqueeze(1) - torch.mm(X,beta)) )/ sigma2
        dlogpri = -beta / sigma2

        dU = (dloglik + dlogpri)

        return dU,sigma2




    def F_constrained(self,x,r_max):
        inactiv_out_max = torch.ones(x.shape[0],1)
        inactiv_out_max[(x.norm(1, dim=-1) - r_max) > 0] = 0

        activ_in = inactiv_out_max
        nabla_bound_eq = torch.sign(x)


        return (self.f_net(x)*activ_in
                - (1-activ_in) * nabla_bound_eq/(nabla_bound_eq.norm(2,dim=-1)[:,None]+0.0001)
                - self.z_net(x)**2 * nabla_bound_eq * activ_in
                )


    def train(self):
        f_optim = optim.Adam(self.f_net.parameters(), lr=f_lr)
        g_optim = optim.Adam(self.g_net.parameters(), lr=f_lr)
        z_optim = optim.Adam(self.z_net.parameters(), lr=f_lr)
        scheduler_f = torch.optim.lr_scheduler.StepLR(f_optim, step_size=3000, gamma=0.9)
        scheduler_g = torch.optim.lr_scheduler.StepLR(g_optim, step_size=3000, gamma=0.9)
        scheduler_z = torch.optim.lr_scheduler.StepLR(z_optim, step_size=3000, gamma=0.9)
        auto_corr = 0.9
        fudge_factor = 1e-6
        historical_grad = 0

        t = torch.linspace(0.1,1,steps=10)
        t = t * ols.norm(p=1)

        beta_plot=torch.zeros(10,D)

        for sf in range(10):

            start = time.time()
            self.particles = torch.randn([num_particle, D])

            r_max = t[sf]

            for ep in range(n_epoch):
                score_target, sigma2 = self.score(self.particles.T)

                self.edge_width = max(self.edge_width / 1.0002, 0.05)

                for i in range(f_iter):
                    self.particles.requires_grad_(True)
                    f_value = self.F_constrained(self.particles,r_max)

                    edge_sample_1 = self.particles[torch.logical_and(torch.norm(self.particles, p=1,dim=-1) < (r_max),
                                                                     torch.norm(self.particles,p=1, dim=-1) > (
                                                                                 r_max - self.edge_width))]

                    weight_1 = (edge_sample_1.shape[0]) / self.particles.shape[0] * 1 / self.edge_width

                    f_optim.zero_grad()
                    g_optim.zero_grad()
                    z_optim.zero_grad()

                    loss_edge_1 = weight_1 * torch.sum(self.F_constrained(edge_sample_1,r_max) * torch.sign(edge_sample_1) / ( (torch.sign(edge_sample_1)).norm(2,dim=-1)[:, None])) \
                                  / edge_sample_1.shape[0] if weight_1 > 0 else 0

                    loss = (-torch.sum(score_target.T * f_value) - torch.sum(
                        self.divergence_approx(f_value, self.particles)) + torch.norm(f_value,
                                                                                      p=p_item) ** p_item / p_item) / \
                           f_value.shape[0] \
                           + loss_edge_1


                    loss.backward()
                    f_optim.step()
                    g_optim.step()
                    z_optim.step()
                    scheduler_f.step()
                    scheduler_g.step()
                    scheduler_z.step()
                    self.particles.requires_grad_(False)
                # update the particle
                with torch.no_grad():
                    gdgrad = self.F_constrained(self.particles,r_max)


                    # use adagrad
                    if ep == 0:
                        historical_grad = historical_grad + gdgrad**2
                    else:
                        historical_grad = auto_corr * historical_grad + (1 - auto_corr) * gdgrad**2
                    adj_grad = (gdgrad)/(fudge_factor + torch.sqrt(historical_grad))
                    self.particles = self.particles + master_stepsize * adj_grad


                    # self.particles = self.particles + master_stepsize * gdgrad




            lasso_pred = torch.median(self.particles, 0).values
            print(lasso_pred,sf)


            beta_plot[sf]=lasso_pred

            error = torch.norm(lasso_pred - ols.squeeze(dim=1), p=1)
            print(error,'l1 error')

            run_time = time.time() - start
            print(run_time,sf)


        #real
        zeros=torch.zeros(1,10)
        beta_plot=torch.cat((zeros, beta_plot), dim = 0)

        beta_plot_save = np.array(beta_plot)

        fig = plt.figure()
        ax = fig.add_subplot()
        for i in range(10):
            ax.plot(torch.linspace(0,1,steps=11), beta_plot[:,i], label=i+1,
                    linestyle='-')

        ax.legend()
        plt.legend(loc="upper right")
        plt.show()




if __name__ == '__main__':
    os.makedirs("result",exist_ok=True)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join("result", 'stdout.txt'))
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(logging.INFO)

    sigma = 1

    f_latent_dim = 256
    f_lr = 0.005
    p_item = 2
    num_particle = 5000
    n_epoch = 300
    f_iter = 10#real
    master_stepsize = 1.2 # real diabetes with adagrad
    edge_width = 1 #real diabetes

    run_ConstrainedGWG = ConstrainedGWG()

    # run_ConstrainedGWG.train() #CFG
    run_ConstrainedGWG.sph_train() #SPH
