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
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import seaborn as sns
sns.set_theme(style="white", color_codes=True, font_scale=1.5)
sns.set_palette("Set2")
import matplotlib.pyplot as plt
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

from collections import defaultdict
from itertools import cycle

from constrained.cube.entropic import UnitCubeEntropic, CubeEntropic
from constrained.sampling.svgd import svgd_update, proj_svgd_update
from constrained.sampling.svmd import svmd_update
from constrained.sampling.kernel import imq
from constrained.target import Target
from ite.cost import BDKL_KnnK
cost = BDKL_KnnK()

K = 1000
D = 2

def run(target, ground_truth_set, method="smvd", lr=0.005):
    theta0 = (np.random.rand(1000, D)* 4 - np.array([2,2])).astype(np.float64)

    theta = theta0
    eta0 = target.mirror_map.nabla_psi(theta)
    # eta: [K, D]
    eta = tf.Variable(eta0)
    n_iters = 2000

    kernel = imq
    eds = []
    kls=[]
    slicews = []

    theta_list = []
    trange = tqdm(range(n_iters))
    optimizer = tf.keras.optimizers.RMSprop(lr)
    for t in trange:
        if method == "svmd":
            eta_grad = svmd_update(target, theta, kernel, n_eigen_threshold=0.998, kernel_width2=0.1, eigen_gpu=True)
        elif method == "msvgd":
            eta_grad = svgd_update(target, eta, theta, kernel, kernel_width2=0.1)
        elif method == "proj_svgd":
            eta_grad = proj_svgd_update(target, eta, theta, kernel, kernel_width2=0.1)
        else:
            raise NotImplementedError()

        optimizer.apply_gradients([(-eta_grad, eta)])
        theta = target.mirror_map.nabla_psi_star(eta)

        if t % 200 == 0:
            theta_list.append(theta.numpy())
    theta_list.append(theta.numpy())
    return theta, eds, theta_list, kls,slicews


class UniformTarget(Target):
    def __init__(self):
        map = UnitCubeEntropic()
        super(UniformTarget, self).__init__(map)

    @tf.function
    def grad_logp(self, theta):
        return tf.zeros_like(theta)

    @tf.function
    def nabla_psi_inv_grad_logp(self, theta):
        return tf.zeros_like(theta)
class Block_corner(Target):
    def __init__(self, shape_param=2, sigma=1):
        map = CubeEntropic(shape_param=shape_param)
        super(Block_corner, self).__init__(map)
        self.shape_param = shape_param
        self.sigma = sigma
        self.mode_mu=-10

    def bondary_eq(self, X):
        return (tf.abs(X[:, 0] - X[:, 1]) + tf.abs(X[:, 0] + X[:, 1])) - self.shape_param ** 2

    def safe_log(self, x):
        return tf.math.log(tf.maximum(tf.constant(1e-128, dtype=x.dtype), x))

    @tf.function
    def grad_logp(self, theta):
        #mog

        tfd = tfp.distributions
        mix = 1 / 9
        var = 0.2
        center = 1.7
        mix_gauss = tfd.Mixture(
            cat=tfd.Categorical(probs=[mix, mix, mix, mix, mix, mix, mix, mix, mix]),
            components=[
                tfd.MultivariateNormalDiag(loc=[-center, center], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[-center, 0], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[-center, -center], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[0, center], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[0, 0], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[0, -center], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[center, center], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[center, 0], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[center, -center], scale_diag=[var, var]),
            ])

        @tf.function
        def mix_log_prob(x: tf.Variable):
            return mix_gauss.log_prob(x)

        theta=tf.cast(theta, tf.float32)

        with tf.GradientTape() as gg:
            gg.watch(theta)
            y = mix_log_prob(theta)
        score_func = gg.gradient(y, theta)

        bond_eq = self.bondary_eq(theta)
        score_ori =score_func[(bond_eq) < 0]
        score_ori = tf.cast(score_ori, tf.float64)

        return score_ori

    #mog
    def sample(self, bs=1000):
        tfd = tfp.distributions
        mix = 1 / 9
        var = 0.2
        # center=1.5
        center = 1.7
        mix_gauss = tfd.Mixture(
            cat=tfd.Categorical(probs=[mix, mix, mix, mix, mix, mix, mix, mix, mix]),
            components=[
                tfd.MultivariateNormalDiag(loc=[-center, center], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[-center, 0], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[-center, -center], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[0, center], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[0, 0], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[0, -center], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[center, center], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[center, 0], scale_diag=[var, var]),
                tfd.MultivariateNormalDiag(loc=[center, -center], scale_diag=[var, var]),
            ])
        X = mix_gauss.sample(50000)
        bond_eq = self.bondary_eq(X)
        return (X[(bond_eq) < 0])[:5000]

###############################################################
target = Block_corner()
rng = np.random.default_rng(12345)

ground_truth_set = tf.cast(target.sample(bs=10000),tf.float64) #load the same target particles as CFG

# plot the target samples
f, ax = plt.subplots(1, 1, figsize=(4, 4))
ground_truth_set_np = ground_truth_set
ax.scatter(ground_truth_set_np[:, 0], ground_truth_set_np[:, 1], alpha=.6, c="g", s=0.5)
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
plt.show()


theta0 = (np.random.rand(1000, D)* 4 - np.array([2,2])).astype(np.float64) #load the same particles as CFG for the same initialization

f, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(theta0[:, 0], theta0[:, 1], alpha=.6, c="g", s=0.5)
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
plt.show()

####################################################################

samples_list_dict = {}
eds_dict = {}
methods = ["msvgd"]
# methods = ["svmd"]
lr_map = {
    "msvgd": 0.05,
    "svmd": 0.01,
}

for method in methods:
    theta, eds,theta_list,kls,slicews = run(target, ground_truth_set, method=method, lr=lr_map[method])
    eds_dict[method] = eds
    samples_list_dict[method] = theta_list

f, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(theta[:, 0], theta[:, 1], alpha=.6, c="g", s=0.5)
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
plt.show()

