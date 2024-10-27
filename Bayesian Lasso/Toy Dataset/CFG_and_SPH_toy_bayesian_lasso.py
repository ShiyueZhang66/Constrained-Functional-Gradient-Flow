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
import torch.distributions as Distributions
from sklearn import preprocessing
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml

import numpy as np
import scipy.io
def density_estimation(m1, m2):
        x_min, x_max = m1.min(), m1.max()
        y_min, y_max = m2.min(), m2.max()
        X, Y = np.mgrid[x_min : x_max : 100j, y_min : y_max : 100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([m1, m2])
        kernel = scipy.stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        return X, Y, Z

device = torch.device(0 if torch.cuda.is_available() else 'cpu')

for seed in ([394]):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    fix_sigma2=25

    D=20
    N=1000

    true_beta=torch.zeros(D,1)
    true_beta[0:10,:]=10

    #load synthetic dataset
    diabetes_X= np.loadtxt('./Lasso_Toy_X.csv', delimiter=',')
    diabetes_y= np.loadtxt('./Lasso_Toy_y.csv', delimiter=',')

    diabetes_X=torch.tensor(diabetes_X,dtype=torch.float32)
    diabetes_y=torch.tensor(diabetes_y,dtype=torch.float32)
    diabetes_y=diabetes_y.unsqueeze(dim=1)

    ols=torch.mm(torch.mm(torch.inverse(torch.mm(diabetes_X.T,diabetes_X)),diabetes_X.T),diabetes_y)
    print(ols)
    print(ols.norm(p=1))

    beta_optimal = torch.mm(torch.mm(torch.inverse(torch.mm(diabetes_X.T, diabetes_X) + torch.eye(D)), diabetes_X.T),
                            diabetes_y)


    print(beta_optimal)
    print(beta_optimal.T)

    # #toy bayesian lasso baseline using rejection sampling
    #
    # m= Distributions.MultivariateNormal(beta_optimal.T, fix_sigma2*torch.inverse(torch.mm(diabetes_X.T, diabetes_X) + torch.eye(D)))
    #
    # beta_sph_mean_convergence=0
    # beta_sph_square_mean_convergence=0
    #
    # count=0
    #
    # Baseline_Samp = torch.zeros(100000, D)
    #
    # while count < 100000:
    #     x_sample=m.sample()
    #     if torch.norm(x_sample, p=1)<ols.norm(p=1):
    #         Baseline_Samp[count] = x_sample
    #
    #         count=count+1
    #         beta_sph_mean_convergence=beta_sph_mean_convergence+x_sample
    #         beta_sph_square_mean_convergence=beta_sph_square_mean_convergence+x_sample**2
    #
    # beta_sph_mean_convergence=beta_sph_mean_convergence/100000
    # beta_sph_square_mean_convergence=beta_sph_square_mean_convergence/100000


    #########################################################################################################################################

    diabetes_y=torch.squeeze(diabetes_y)

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

        def potential(self, theta, ts, sigma2, y=diabetes_y, X=diabetes_X):
            K = theta.size(0)
            beta = torch.abs(theta) * theta
            beta = beta[0:K - 1] * ts
            loglik = -torch.norm(y.unsqueeze(1) - torch.mm(X, beta)) ** 2 / (2 * sigma2)
            logpri = -torch.norm(beta) ** 2 / (2 * sigma2)
            U = -(loglik + logpri)
            return U

        def d_potential(self, theta, ts, sigma2, y=diabetes_y, X=diabetes_X):
            K = theta.size(0)
            beta = torch.abs(theta) * theta
            beta = beta[0:K - 1] * ts
            dloglik = torch.mm(X.T, (y.unsqueeze(1) - torch.mm(X, beta))) / sigma2
            dlogpri = -beta / sigma2
            dU = -(dloglik + dlogpri) * ts * 2 * torch.abs(theta[0:K - 1])
            dUt = torch.cat([dU, torch.tensor(0).unsqueeze(dim=0).unsqueeze(dim=1)]) - theta * torch.mm(theta[0:K - 1].T, dU)
            return dUt

        def sph_hmc(self,current_q,ts,sigma2,eps=0.2, L=5):
            length = current_q.size(0)
            q = current_q
            tst=ts
            # sample velocity
            v = torch.randn((length,1))  # standard multinormal
            q=torch.reshape(q,(length,1))
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
            # the position at the end of the trajectory or the initial position  MARK THAT!!!
            logAP = -proposed_E + current_E

            if logAP > min(0, torch.log(torch.rand(1))):
                Ind = 1
                return (q, Ind)
            else:
                Ind = 0
                return (current_q, Ind)

        #CFG
        def score(self, beta, number,y=diabetes_y, X=diabetes_X,sigma2=100, t=0):

            beta_randint = torch.randint(0, number, (1,))
            beta_pred = beta[:, beta_randint]

            var=torch.norm(y.unsqueeze(1) - torch.mm(X,beta_pred))**2+torch.norm(beta_pred)**2
            sigma2=1/torch.distributions.Gamma((N-1+D)/2,var/2).sample()

            sigma2 = fix_sigma2

            dloglik = torch.mm(X.T,(y.unsqueeze(1) - torch.mm(X,beta)) )/ sigma2
            dlogpri = -beta / sigma2
            dU = (dloglik + dlogpri)

            return dU,sigma2


        def bondary_eq(self,X,r_max):
            return torch.norm(X,p=1,dim=-1)-r_max

        def nabla_bound(self,X):
            return torch.sign(X)

        def F_constrained(self,x,r_max):
            inactiv_out_max = torch.ones(x.shape[0],1)
            inactiv_out_max[(x.norm(1, dim=-1) - r_max) > 0] = 0

            activ_in = inactiv_out_max
            nabla_bound_eq = torch.sign(x)

            return (self.f_net(x)*activ_in
                    - (1-activ_in) * nabla_bound_eq/(nabla_bound_eq.norm(2,dim=-1)[:,None]+0.0001)
                    # - 1000*(1 - activ_in) * nabla_bound_eq / (nabla_bound_eq.norm(2, dim=-1)[:, None] + 0.0001)
                    - self.z_net(x)**2 * nabla_bound_eq * activ_in
                    )

        def overall_train(self):
            equal_sample_num = [5,15,30,70,180,400,900,2000]
            equal_sample_num_length=len(equal_sample_num)

            for sample_num_id in range(equal_sample_num_length):

                self.f_net = F_net(D, f_latent_dim).to(self.device)
                self.g_net = G_net(D, f_latent_dim).to(self.device)
                self.z_net = G_net(D, f_latent_dim).to(self.device)

                NSamp = equal_sample_num[sample_num_id]
                NBurnIn =4000
                # NBurnIn = 8000
                NSamp=NSamp+NBurnIn
                Nsf = 10

                #cfg
                f_optim = optim.Adam(self.f_net.parameters(), lr=f_lr)
                g_optim = optim.Adam(self.g_net.parameters(), lr=f_lr)
                z_optim = optim.Adam(self.z_net.parameters(), lr=f_lr)
                scheduler_f = torch.optim.lr_scheduler.StepLR(f_optim, step_size=3000, gamma=0.9)
                scheduler_g = torch.optim.lr_scheduler.StepLR(g_optim, step_size=3000, gamma=0.9)
                scheduler_z = torch.optim.lr_scheduler.StepLR(z_optim, step_size=3000, gamma=0.9)
                auto_corr = 0.9
                fudge_factor = 1e-6
                historical_grad = 0
                beta_plot = torch.zeros(10, D)

                ###################################################################

                t = torch.linspace(0.1, 1, steps=10)
                t = t * ols.norm(p=1)

                idex = [9]
                for sf in idex:

    ################# sph unlock the block below ##############################################################################################

                    #sph
                    Samp = torch.zeros(NSamp - NBurnIn, D)
                    TrjLength = torch.zeros(1000)
                    Stepsz = torch.zeros(1000)
                    wt = torch.zeros(NSamp - NBurnIn)
                    start = time.time()

                    ## HMC setting
                    TrjLength[sf] = 2 * math.pi / D * (1 - (sf + 1) / Nsf / 4 * 3)
                    NLeap = 100
                    # NLeap=50

                    Stepsz[sf] = TrjLength[sf] / NLeap

                    ## Initialization
                    beta = torch.zeros(D) / t[sf]

                    theta = torch.sign(beta) * torch.sqrt(torch.abs(beta))
                    theta = torch.cat([theta, torch.sqrt(1 - torch.norm(theta) ** 2).unsqueeze(dim=0)])

                    beta = beta.unsqueeze(1)
                    var = torch.norm(diabetes_y.unsqueeze(1) - torch.mm(diabetes_X, beta)) ** 2 + torch.norm(beta) ** 2
                    sigma2 = 1 / torch.distributions.Gamma((N - 1 + D) / 2, var / 2).sample()

                    sigma2 =fix_sigma2

                    for Iter in range(NSamp):

                        # Use Spherical HMC to get sample theta
                        samp, Ind = self.sph_hmc(theta, t[sf], sigma2, Stepsz[sf], NLeap)
                        theta = samp
                        # accp = accp + samp$Ind
                        beta = torch.abs(theta) * theta
                        beta = beta[0:D] * t[sf]
                        beta = torch.reshape(beta, (D, 1))

                        # sample sigma2
                        var = torch.norm(diabetes_y.unsqueeze(1) - torch.mm(diabetes_X, beta)) ** 2 + torch.norm(beta) ** 2
                        sigma2 = 1 / torch.distributions.Gamma((N - 1 + D) / 2, var / 2).sample()

                        sigma2 = fix_sigma2

                        # save sample beta
                        if Iter + 1 > NBurnIn:
                            beta = beta.squeeze(dim=1)
                            Samp[Iter - NBurnIn] = beta
                            wt[Iter - NBurnIn] = torch.log((2 * t[sf])) * D + torch.sum(torch.log(torch.abs(theta)))

                    max_exp = torch.max(wt)
                    wt = torch.exp(wt - max_exp)

                    # Resample
                    sampler = list(WeightedRandomSampler(wt, NSamp - NBurnIn, replacement=True))
                    set_sampler = set(sampler)
                    print(len(set_sampler))
                    print(len(sampler))
                    print(set_sampler)
                    print(sampler)


                    distinct_num=np.array([1])
                    distinct_num[0]=len(set_sampler)

                    ReSamp = Samp[sampler]  # Resample
                    # ReSamp = Samp# non-Resample, inaccurate
                    ReSamp = ReSamp.clone().detach()

                    #save
                    ReSamp_save = np.array(ReSamp)

                    from sklearn.neighbors import KernelDensity
                    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
                    kde.fit(ReSamp[:, 0:2])
                    kde_logprob = kde.score_samples(ReSamp[:, 0:2])

                    mean_ReSamp=torch.mean(ReSamp,dim=0)
                    print(mean_ReSamp,'mean_resamp')
                    sq_mean_ReSamp = torch.mean(ReSamp**2, dim=0)
                    print(sq_mean_ReSamp, 'sq_resamp')

#################### cfg unlock the block below ##############################################################################################

    #                 ##############################################################################################################
    #
                    #cfg

                    num_particle=equal_sample_num[sample_num_id]


                    start = time.time()

                    self.particles = torch.randn([num_particle, D]) * 0.3  # synthetic

                    r_max = t[sf]

                    for ep in range(n_epoch):

                        score_target, sigma2_2 = self.score(self.particles.T,number=num_particle)
                        self.edge_width=num_particle**(-1/3)/10 #adaptive edgewidth scheme

                        for i in range(f_iter):
                            self.particles.requires_grad_(True)
                            f_value = self.F_constrained(self.particles, r_max)

                            edge_sample_1 = self.particles[
                                torch.logical_and(torch.norm(self.particles, p=1, dim=-1) < (r_max + self.edge_width),
                                                  torch.norm(self.particles, p=1, dim=-1) > (
                                                          r_max - self.edge_width))]

                            # self.edge_sample_ind = (self.bondary_eq(
                            #     self.particles - torch.sign(self.bondary_eq(self.particles,r_max)[:,
                            #                                 None]) * self.edge_width * self.nabla_bound(
                            #         self.particles) / (self.nabla_bound(self.particles).norm(2, dim=-1)[:, None])
                            # ,r_max)>0)* (self.bondary_eq(self.particles,r_max) < 0)
                            # edge_sample_1 = self.particles[self.edge_sample_ind]

                            # edge_sample_1 = self.particles[
                            #     torch.logical_and(torch.norm(self.particles, p=1, dim=-1) < (r_max),
                            #                       torch.norm(self.particles, p=1, dim=-1) > (
                            #                               r_max - (20.0)**(1/2)*self.edge_width))]

                            weight_1 = (edge_sample_1.shape[0]) / self.particles.shape[0] * 1 / self.edge_width

                            f_optim.zero_grad()
                            z_optim.zero_grad()

                            loss_edge_1 = weight_1 * torch.sum(
                                self.F_constrained(edge_sample_1, r_max) * torch.sign(edge_sample_1) / (
                                (torch.sign(edge_sample_1)).norm(2, dim=-1)[:, None])) \
                                          / edge_sample_1.shape[0] if weight_1 > 0 else 0

                            loss = (-torch.sum(score_target.T * f_value) - torch.sum(
                                self.divergence_approx(f_value, self.particles)) + torch.norm(f_value,
                                                                                              p=p_item) ** p_item / p_item) / \
                                   f_value.shape[0] \
                                   + loss_edge_1


                            loss.backward()
                            f_optim.step()
                            z_optim.step()
                            scheduler_f.step()
                            scheduler_z.step()
                            self.particles.requires_grad_(False)
                        # update the particle
                        with torch.no_grad():
                            gdgrad = self.F_constrained(self.particles, r_max)

                            self.particles = self.particles + master_stepsize * gdgrad

                            # # adagrad, not needed for this paper
                            # if ep == 0:
                            #     historical_grad = historical_grad + gdgrad**2
                            # else:
                            #     historical_grad = auto_corr * historical_grad + (1 - auto_corr) * gdgrad**2
                            # adj_grad = (gdgrad)/(fudge_factor + torch.sqrt(historical_grad))
                            # # adj_grad = f_net(self.particle_weight)
                            # self.particles = self.particles + master_stepsize * adj_grad

                        if (ep+1)%50==0:
                            end = time.time()
                            print(end - start, f'total_time_{seed}_{ep+1}')

                            # #save
                            # particles_in_save = np.array(particles_in)



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

        f_latent_dim = 256
        f_lr = 0.0005
        p_item = 2
        n_epoch = 1000
        f_iter = 10
        master_stepsize = 0.004
        edge_width = 0.1

        run_ConstrainedGWG = ConstrainedGWG()
        run_ConstrainedGWG.overall_train()

