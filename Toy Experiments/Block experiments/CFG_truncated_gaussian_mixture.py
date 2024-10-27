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
from target_models_2d import Block,Block_corner,Block_mirror_hard,Block_mirror_harder,Block_nineMG,Block_nineMG_edge
import matplotlib.pyplot as plt
import pdb
import ot


seed = 1214
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


from ite.cost import BDKL_KnnK
cost = BDKL_KnnK()

class F_net(nn.Module):
    def __init__(self, z_dim, latent_dim = 128):
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
    def __init__(self, z_dim, latent_dim = 128):
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
        self.model = Block_nineMG_edge(self.device, sigma=sigma, shape_param=shape_param)
        self.f_net = F_net(2,f_latent_dim).to(self.device)
        self.g_net = G_net(2,f_latent_dim).to(self.device)
        self.z_net = G_net(2,f_latent_dim).to(self.device)
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
    def F_constrained(self,x):
        activ_in = torch.ones(x.shape[0],1)
        activ_in[ self.model.bondary_eq(x) > 0] = 0 
        nabla_bound_eq = self.model.nabla_bound(x)
        
        active_edge = torch.zeros(x.shape[0],1)
        edge_x_ind = self.model.bondary_eq(
                    x - torch.sign(self.model.bondary_eq(x)[:,None])*self.edge_width * self.model.nabla_bound(x)/(self.model.nabla_bound(x).norm(2,dim=-1)[:,None])
                    ) * self.model.bondary_eq(x) < 0
        
        active_edge[edge_x_ind] = 1
        return (self.f_net(x) * activ_in 
                # - self.z_net(x)**2 * nabla_bound_eq/(nabla_bound_eq.norm(2,dim=-1)[:,None]+0.0001)
                - (1-activ_in) * nabla_bound_eq/(nabla_bound_eq.norm(2,dim=-1)[:,None]+0.0001)
                - (self.z_net(x)**2).clamp(max=1) * nabla_bound_eq * activ_in # with inner znet
                # - (self.z_net(x)**2).clamp(max=0.1) * nabla_bound_eq * active_edge # boundary znet
                )
    def train(self):

        for sample_numb in ([1000]):

            init_mu = 0
            init_var = "uniform"
            self.particles = torch.rand([sample_numb * 10, 2]) * 4- torch.tensor([2, 2])

            bond_eq_init = self.model.bondary_eq(self.particles)
            self.particles=(self.particles[(bond_eq_init) < 0])[:sample_numb]

            f_optim = optim.Adam(self.f_net.parameters(), lr=f_lr)
            g_optim = optim.Adam(self.g_net.parameters(), lr=f_lr)
            z_optim = optim.Adam(self.z_net.parameters(), lr=f_lr)
            scheduler_f = torch.optim.lr_scheduler.StepLR(f_optim, step_size=3000, gamma=0.9)
            scheduler_g = torch.optim.lr_scheduler.StepLR(g_optim, step_size=3000, gamma=0.9)
            scheduler_z = torch.optim.lr_scheduler.StepLR(z_optim, step_size=3000, gamma=0.9)
            auto_corr = 0.9
            fudge_factor = 1e-6
            historical_grad = 0

            for ep in range(n_epoch):
                score_target = self.model.score(self.particles)
                self.particles=self.particles.detach()

                self.edge_sample_ind = self.model.bondary_eq(
                        self.particles - torch.sign(self.model.bondary_eq(self.particles)[:,None])*self.edge_width * self.model.nabla_bound(self.particles)/(self.model.nabla_bound(self.particles).norm(2,dim=-1)[:,None])
                        ) * self.model.bondary_eq(self.particles) < 0
                edge_sample = self.particles[self.edge_sample_ind]

                # if ep % 2000 == 0 or ep == (n_epoch-1):
                if ep % 20 == 0 or ep == (n_epoch - 1):
                    fig, ax = plt.subplots(figsize=(5, 5))
                    self.model.contour_plot(ax, fnet = self.F_constrained, samples = [self.particles,edge_sample], save_to_path="result/t_{}_{}_no_inner_width_{}_stepsize_{}_seed_{}_sharp_10_bd_epoch_{}_f_iter_{}_f_lr_{}_f_latent_dim_{}_{}_{}_20_itergap_no_bd_no_znet.png".format(ep,sample_numb,edge_width,master_stepsize,seed,n_epoch,f_iter,f_lr,f_latent_dim,init_mu,init_var),fig_title='Iter: {}'.format(ep),quiver=True)
                    plt.close()

                self.edge_width = max(self.edge_width/1.0002,0.05)

                for i in range(f_iter):
                    self.particles.requires_grad_(True)
                    f_value = self.F_constrained(self.particles)
                    # edge_sample = self.particles[torch.abs(self.model.bondary_eq(self.particles)) < self.edge_width/2]
                    weight = (edge_sample.shape[0])/self.particles.shape[0] * 1/self.edge_width

                    f_optim.zero_grad()
                    g_optim.zero_grad()
                    z_optim.zero_grad()
                    loss_edge = weight * torch.sum(self.F_constrained(edge_sample) * \
                                                       self.model.nabla_bound(edge_sample)/self.model.nabla_bound(edge_sample).norm(2,dim=-1)[:,None])\
                                                        /edge_sample.shape[0] if weight > 0 else 0
                    loss = (-torch.sum(score_target * f_value) - torch.sum(self.divergence_approx(f_value, self.particles)) + torch.norm(f_value, p=p_item)**p_item /p_item)/f_value.shape[0] \
                        + loss_edge

                    # no boundary integral
                    # loss = (-torch.sum(score_target * f_value) - torch.sum(
                    #     self.divergence_approx(f_value, self.particles)) + torch.norm(f_value,
                    #                                                                   p=p_item) ** p_item / p_item) / \
                    #        f_value.shape[0]

                    loss.backward()
                    f_optim.step()
                    scheduler_f.step()
                    if i == 0:
                        z_optim.step()
                        scheduler_z.step()
                    g_optim.step()
                    scheduler_g.step()
                    self.particles.requires_grad_(False)
                # update the particle
                with torch.no_grad():
                    gdgrad = self.F_constrained(self.particles)
                    self.particles = self.particles + master_stepsize * gdgrad

                if ep % 100 == 0 or ep == (n_epoch-1):
                    with torch.no_grad():
                        logging.info("f_net: {:.4f}, z_net: {:.4f}".format(torch.abs(self.f_net(self.particles)).mean(),torch.abs(self.z_net(self.particles)).mean()))
                    logging.info("Ep: {}, loss: {:.4f}, loss_edge: {:.4f}, ratio: {:.4f} weight :{:.4f}".format(ep, loss, loss_edge, loss_edge/loss,weight))



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
    shape_param = 2
    f_latent_dim = 128
    f_lr = 0.002
    # f_lr = 0.0005
    p_item = 2
    num_particle = 1000
    n_epoch = 2020
    f_iter = 10

    # master_stepsize = 0.0005
    # master_stepsize = 0.00001
    master_stepsize = 0.001
    # master_stepsize = 0.01

    # edge_width = 0.1
    # edge_width = 0.01
    # edge_width = 0.003
    # edge_width = 0.0005
    edge_width = 0.001
    
    run_ConstrainedGWG = ConstrainedGWG()
    run_ConstrainedGWG.train()
