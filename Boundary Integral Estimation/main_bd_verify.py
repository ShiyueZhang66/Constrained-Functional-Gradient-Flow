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
from target_models_2d import Block,Block_corner
import matplotlib.pyplot as plt
import pdb


# For Appendix C

class ConstrainedGWG(object):
    def __init__(self, args=None):
        self.args = args
        self.device = "cpu"
        # self.model = Block_corner(self.device, sigma=sigma,shape_param=shape_param) #p3
        self.model = Block(self.device, sigma=sigma, shape_param=shape_param) #p1,p2
        self.edge_width = edge_width

    def velo(self,X):
        return self.model.nabla_bound(X) / self.model.nabla_bound(X).norm(2,dim=-1)[:, None] #v1
        # return torch.cat((torch.unsqueeze(X[:,1],1),torch.unsqueeze(X[:,0],1)),1) #v2
        # return torch.cat((torch.unsqueeze(X[:, 1]**2, 1), torch.unsqueeze(X[:, 0]**2, 1)), 1) #v3




    def train(self):
        # for p1
        # gt_samples = self.model.sample(bs=10)
        gt_samples = self.model.sample(bs=100)
        # gt_samples = self.model.sample(bs=1000)
        # gt_samples = self.model.sample(bs=10000)
        # gt_samples = self.model.sample(bs=100000)




        # for p2,p3
        # gt_samples = self.model.sample(bs=100)
        # gt_samples = self.model.sample(bs=1000)
        # gt_samples = self.model.sample(bs=10000)
        # gt_samples = self.model.sample(bs=100000)
        # gt_samples = self.model.sample(bs=1000000)




        plt.scatter(gt_samples[:, 0], gt_samples[:, 1])
        plt.show()

        # self.edge_width = 0.05 * 10 ** (1 / 3)
        self.edge_width = 0.05
        # self.edge_width = 0.05*10**(-1/3)
        # self.edge_width = 0.05*10**(-2/3)
        # self.edge_width = 0.005


        gt_edge_sample_ind = self.model.bondary_eq(
            gt_samples - torch.sign(
                self.model.bondary_eq(gt_samples)[:, None]) * self.edge_width * self.model.nabla_bound(
                gt_samples) / (self.model.nabla_bound(gt_samples).norm(2, dim=-1)[:, None])
        ) * self.model.bondary_eq(gt_samples) < 0
        gt_edge_sample = gt_samples[gt_edge_sample_ind]

        weight = (gt_edge_sample.shape[0]) / gt_samples.shape[0] * 1 / self.edge_width

        loss_edge = weight * torch.sum(self.velo(gt_edge_sample) * \
                                       self.model.nabla_bound(gt_edge_sample) / self.model.nabla_bound(gt_edge_sample).norm(2,
                                                                                                                      dim=-1)[
                                                                             :, None]) \
                    / gt_edge_sample.shape[0] if weight > 0 else 0

        gt_loss_edge =loss_edge

        print(gt_loss_edge,'estimated boundary integral')




if __name__ == '__main__':

    sigma = 1
    shape_param = 2
    p_item = 2
    edge_width=0.05

    run_ConstrainedGWG = ConstrainedGWG()
    run_ConstrainedGWG.train()
