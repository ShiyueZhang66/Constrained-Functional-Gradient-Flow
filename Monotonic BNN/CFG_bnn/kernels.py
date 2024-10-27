
import torch
import numpy as np


class Gaussian_kernel(torch.nn.Module):
    def __init__(self, h=-1):
        super(Gaussian_kernel, self).__init__()
        self.h = h

    def kxy(self, samples_x, samples_y):
        '''
        input:
            samples_x: shape = [n,d]
            samples_y: shape = [n,d]
        return:
            kxy: shape = [n,n]
        '''
        pairwise_dists = (
            (samples_x[:, None, :] - samples_y[None, :, :])**2).sum(-1)
        if self.h < 0:  # use the median trick
            with torch.no_grad():
                h = torch.median(pairwise_dists)
                h = torch.sqrt(0.5 * h / np.log(samples_x.shape[0] + 1))
        else:
            h = h
        kxy = torch.exp(- pairwise_dists / (2 * h**2 + 1e-8))
        return kxy

    def dkxy_sum(self, samples_x, samples_y):
        '''
        input:
            samples_x: shape = [n,d]
            samples_y: shape = [n,d]
        return:
            dkxy_sum_x: shape = [n,d]
            dkxy_sum_y: shape = [n,d]
        '''
        xy = samples_x[:, None, :] - samples_y[None, :, :]
        pairwise_dists = ((xy)**2).sum(-1)
        if self.h < 0:  # use the median trick
            with torch.no_grad():
                h = torch.median(pairwise_dists)
                h = torch.sqrt(0.5 * h / np.log(samples_x.shape[0] + 1))
        else:
            h = h
        kxy = torch.exp(- pairwise_dists / (2 * h**2 + 1e-8))
        dkxy_x = - kxy[:, :, None] * xy / (h**2 + 0.5 * 1e-8)
        dkxy_sum_x = torch.sum(dkxy_x, dim=0)
        dkxy_sum_y = torch.sum(dkxy_x, dim=1)
        return dkxy_sum_x, dkxy_sum_y
