import numpy as np
import torch
from torch import optim
import logging
import os
from scipy.spatial.distance import pdist, squareform
from model_bnn import BNNClassifier, F_net, Z_net
from projector.dynamic_barrier import DynamicBarrier
from utils import load_data
from kernels import Gaussian_kernel
import math


class RUNNER(object):

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = BNNClassifier(self.device,
                                   config.data.input_dim,
                                   n_hidden=config.model.n_hidden)
        self.path = config.path
        weight_dim = (config.data.input_dim + 1) * config.model.n_hidden + (
            config.model.n_hidden + 1) * config.data.output_dim + 1
        self.particles = torch.randn([config.training.n_particle,
                                      weight_dim]).to(self.device)

    def get_datasets(self):
        dataset = self.config.data.dataset
        try:
            self.X_train, self.y_train, self.X_test, self.y_test = load_data(
                dataset)
        except:
            self.X_train, self.y_train, self.z_train, self.X_test, self.y_test, self.z_test = load_data(
                dataset)
            self.z_train = torch.from_numpy(
                self.z_train).float().to(self.device)
            self.z_test = torch.from_numpy(self.z_test).float().to(self.device)

        self.X_train = torch.from_numpy(self.X_train).float().to(self.device)
        self.y_train = torch.from_numpy(self.y_train).float().to(
            self.device).unsqueeze(1)
        self.X_test = torch.from_numpy(self.X_test).float().to(self.device)
        self.y_test = torch.from_numpy(self.y_test).float().to(
            self.device).unsqueeze(1)

        return self.X_train, self.y_train, self.X_test, self.y_test

    def init_weights(self, X_train, y_train):
        w1 = 1.0 / np.sqrt(X_train.shape[1] + 1) * torch.randn(
            self.config.training.n_particle,
            X_train.shape[1] * self.config.model.n_hidden).to(self.device)
        b1 = torch.zeros((self.config.training.n_particle,
                          self.config.model.n_hidden)).to(self.device)
        w2 = 1.0 / np.sqrt(self.config.model.n_hidden + 1) * torch.randn(
            self.config.training.n_particle, self.config.model.n_hidden).to(
                self.device)
        b2 = torch.zeros((self.config.training.n_particle, 1)).to(self.device)

        loglambda = torch.ones(
            (self.config.training.n_particle, 1)).to(self.device) * np.log(
                np.random.gamma(1, 0.1))
        return torch.cat([w1, b1, w2, b2, loglambda], dim=1).to(self.device)

    def boundary_eq(self, Z):
        if Z.shape[0] == 0:
            return None
        if self.config.model.loss_type == "mono":
            return self.model.mono_loss(Z, self.X_train,
                                        self.config.model.mono_index,
                                        self.config.model.target_ub)
        elif self.config.model.loss_type == "fair":
            return self.model.fair_loss(Z, self.X_train, self.z_train,
                                        self.config.model.target_ub)

    def nabla_boundary_eq(self, Z):
        if Z.shape[0] == 0:
            return None
        if self.config.model.loss_type == "mono":
            return self.model.nabla_mono_loss(Z, self.X_train,
                                              self.config.model.mono_index,
                                              self.config.model.target_ub)
        elif self.config.model.loss_type == "fair":
            return self.model.nabla_fair_loss(Z, self.X_train,
                                              self.z_train,
                                              self.config.model.target_ub)

    def boundary_eq_test(self, Z):
        if Z.shape[0] == 0:
            return None
        if self.config.model.loss_type == "mono":
            return self.model.mono_loss(Z, self.X_test,
                                        self.config.model.mono_index,
                                        self.config.model.target_ub)
        elif self.config.model.loss_type == "fair":
            return self.model.fair_loss(Z, self.X_test, self.z_test,
                                        self.config.model.target_ub)


class ConstrainedGWG(RUNNER):

    def __init__(self, config):
        super().__init__(config)
        weight_dim = (config.data.input_dim + 1) * config.model.n_hidden + (
            config.model.n_hidden + 1) * config.data.output_dim + 1
        self.f_net = F_net(weight_dim,
                           config.model.f_latent_dim).to(self.device)
        self.z_net = Z_net(weight_dim,
                           config.model.z_latent_dim).to(self.device)
        self.edge_width = config.training.edge_width
        self.load_path = config.load_path

    def divergence_approx(self, fnet_value, parti_input, e=None):

        def sample_rademacher_like(y):
            return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1

        if e is None:
            e = sample_rademacher_like(parti_input)
        e_dzdx = torch.autograd.grad(fnet_value,
                                     parti_input,
                                     e,
                                     create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = e_dzdx_e.view(parti_input.shape[0], -1).sum(dim=1)
        return approx_tr_dzdx

    def F_constrained(self, x, use_threshold=True):

        if not use_threshold:
            nabla_bound_eq = self.nabla_boundary_eq(x.detach())
            return self.f_net(x) - (4 * self.z_net(x)**2).clamp(max=10) * nabla_bound_eq
        else:
            bound_eq = self.boundary_eq(x.detach())
            nabla_bound_eq = self.nabla_boundary_eq(x.detach())
            activ_in = torch.ones(x.shape[0], 1, device=self.device)
            activ_in[bound_eq > 0] = 0

            # return ((self.f_net(x) -
            #         (4 * self.z_net(x)**2).clamp(max=10) * nabla_bound_eq) *
            #         activ_in - (1 - activ_in) * 100 * nabla_bound_eq /
            #         (nabla_bound_eq.norm(2, dim=-1)[:, None] + 1e-6)) # for mono

            return (self.f_net(x) *
                    activ_in - (1 - activ_in) * 100 * nabla_bound_eq /
                    (nabla_bound_eq.norm(2, dim=-1)[:, None] + 1e-6))  # for mono ablation

            # # for fair
            # return ((self.f_net(x) - (4 * self.z_net(x)**2).clamp(max=10) * nabla_bound_eq) * activ_in - (1 - activ_in) * 10 * nabla_bound_eq / (nabla_bound_eq.norm(2, dim=-1)[:, None] + 1e-6))

    def train(self):
        X_train, y_train, X_test, y_test = self.get_datasets()
        self.scale_sto = X_train.shape[0] / self.config.training.batch_size

        self.particles = self.init_weights(X_train, y_train)

        f_optim = optim.Adam(self.f_net.parameters(),
                             lr=self.config.optim.f_lr)
        z_optim = optim.Adam(self.z_net.parameters(),
                             lr=self.config.optim.z_lr)
        scheduler_f = torch.optim.lr_scheduler.StepLR(f_optim,
                                                      step_size=1000,
                                                      gamma=0.9)
        scheduler_z = torch.optim.lr_scheduler.StepLR(z_optim,
                                                      step_size=1000,
                                                      gamma=0.9)

        if self.load_path != "":
            state_dict = torch.load(os.path.join(
                self.load_path, "checkpoint.pth"))
            self.particles = state_dict["particle"].to(self.device)
            self.f_net.load_state_dict(state_dict["f_net"])
            self.z_net.load_state_dict(state_dict["z_net"])
            f_optim.load_state_dict(state_dict["f_optim"])
            z_optim.load_state_dict(state_dict["z_optim"])
            self.edge_width = state_dict["edge_width"]

        use_threshold = True  # for mono
        # use_threshold = False  # for fair

        info = {'test_acc': [], 'test_llk': [], 'target_loss_test': [],
                'target_loss': [], 'train_llk': [], "ratio_out": []}

        for ep in range(self.config.training.n_iter):
            N0 = X_train.shape[0]
            batch = [
                i % N0
                for i in range(ep * self.config.training.batch_size, (ep + 1) *
                               self.config.training.batch_size)
            ]
            x = X_train[batch]
            y = y_train[batch]

            score_target = self.model.score(self.particles, x, y,
                                            self.scale_sto)
            bound_eq = self.boundary_eq(self.particles)
            nabla_bound_eq = self.nabla_boundary_eq(self.particles)
            self.edge_sample_ind = self.boundary_eq(
                self.particles - torch.sign(bound_eq[:, None]) *
                self.edge_width * nabla_bound_eq /
                (nabla_bound_eq.norm(2, dim=-1)[:, None])) * bound_eq < 0
            edge_sample = self.particles[self.edge_sample_ind]
            self.edge_width = self.edge_width / 1.002

            f_iter = self.config.optim.f_iter

            for i in range(f_iter):
                self.particles.requires_grad_(True)

                f_value = self.F_constrained(self.particles, use_threshold)

                edge_nabla_bound_eq = self.nabla_boundary_eq(edge_sample)

                weight = (edge_sample.shape[0]
                          ) / self.particles.shape[0] / self.edge_width

                f_optim.zero_grad()
                z_optim.zero_grad()

                if use_threshold:
                    loss_edge = weight * torch.sum(
                        self.F_constrained(edge_sample) * edge_nabla_bound_eq /
                        edge_nabla_bound_eq.norm(2, dim=-1)[:, None]
                    ) / edge_sample.shape[0] if weight > 0 else 0

                    loss = (-torch.sum(score_target * f_value) - torch.sum(
                        self.divergence_approx(f_value, self.particles)) +
                        torch.norm(f_value, p=2)**2 /
                        2) / f_value.shape[0] + loss_edge
                else:
                    loss = (-torch.sum(score_target * f_value) - torch.sum(
                        self.divergence_approx(f_value, self.particles)) +
                        torch.norm(f_value, p=2)**2 /
                        2) / f_value.shape[0]
                loss.backward()
                f_optim.step()
                scheduler_f.step()
                z_optim.step()
                scheduler_z.step()

            # update the particle
            gdgrad = self.F_constrained(self.particles, use_threshold)
            with torch.no_grad():
                self.particles = self.particles + self.config.training.particle_lr * gdgrad

            if ep % self.config.training.checkpoint_frq == 0 or ep == (
                    self.config.training.n_iter - 1):
                # with torch.no_grad():
                #     logging.info("f_net: {:.4f}, z_net: {:.4f}".format(
                #         torch.abs(self.f_net(self.particles)).mean(),
                #         torch.abs(self.z_net(self.particles)).mean()))
                # logging.info(
                #     "Ep: {}, loss: {:.4f}, loss_edge: {:.4f}, ratio: {:.4f} weight :{:.4f}"
                #     .format(ep, loss, loss_edge, loss_edge / loss, weight))
                train_acc, train_llk = self.model.bce_llk(
                    self.particles, X_train, y_train)
                logging.info("train acc: {:.3f}, train llk: {:.4f}".format(
                    train_acc, train_llk))
                test_acc, test_llk = self.model.bce_llk(
                    self.particles, X_test, y_test)
                # if train_acc > 0.82:
                #     use_threshold = True
                # print(use_threshold)
                bound_eq = self.boundary_eq(self.particles)
                ratio_out = np.sum(bound_eq.detach().cpu().numpy() >
                                   0) / self.config.training.n_particle
                target_loss = bound_eq.mean() + self.config.model.target_ub
                target_loss_test = self.boundary_eq_test(
                    self.particles).mean() + self.config.model.target_ub
                logging.info(
                    "Ep: {}, test acc: {:.3f}, test llk: {:.4f}, target loss: {:.4f}, ratio_out: {:.4f}, test target loss: {:.4f}"
                    .format(ep, test_acc, test_llk, target_loss, ratio_out,
                            target_loss_test))
                state_dict = {
                    "particle": self.particles.detach().cpu(),
                    "f_net": self.f_net.state_dict(),
                    "z_net": self.z_net.state_dict(),
                    "f_optim": f_optim.state_dict(),
                    "z_optim": z_optim.state_dict(),
                    "edge_width": self.edge_width
                }
                info["test_acc"] += [test_acc]
                info["test_llk"] += [test_llk]
                info["target_loss_test"] += [target_loss_test]
                info["target_loss"] += [target_loss]
                info["train_llk"] += [train_llk]
                info["ratio_out"] += [ratio_out]

                torch.save(info,
                           os.path.join(self.path, "info.pth"))
                torch.save(state_dict,
                           os.path.join(self.path, "checkpoint.pth"))
        return info


class PrimalDualSVGD(RUNNER):

    def __init__(self, config):
        super().__init__(config)
        self.lbd_lr = config.optim.lambda_lr

    def svgd_kernel(self, x, h=-1):
        x = x.detach().numpy()
        sq_dist = pdist(x)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(x.shape[0] + 1))

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, x)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(x.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(x[:, i], sumkxy)
        dxkxy = dxkxy / (h**2)
        return torch.from_numpy(Kxy).float().to(self.device), torch.from_numpy(dxkxy).float().to(self.device)

    def train(self):
        X_train, y_train, X_test, y_test = self.get_datasets()
        self.scale_sto = X_train.shape[0] / self.config.training.batch_size
        self.particles = self.init_weights(X_train, y_train)
        lbd = 10.0
        # lbd = 1000.0

        info = {'test_acc': [], 'test_llk': [], 'target_loss_test': [],
                'target_loss': [], ' train_llk': [], "ratio_out": []}

        for ep in range(self.config.training.n_iter):
            N0 = X_train.shape[0]
            batch = [
                i % N0
                for i in range(ep * self.config.training.batch_size, (ep + 1) *
                               self.config.training.batch_size)
            ]
            x = X_train[batch]
            y = y_train[batch]

            score_target = self.model.score(self.particles, x, y,
                                            self.scale_sto)
            bound_eq = self.boundary_eq(self.particles)
            nabla_bound_eq = self.nabla_boundary_eq(self.particles)
            lbd = max(lbd + self.lbd_lr * bound_eq.mean(), 0)

            # update the particle
            with torch.no_grad():
                kxy, dxkxy = self.svgd_kernel(self.particles.cpu(), h=-1)
                # s = score_target - lbd * nabla_bound_eq
                s = score_target  # svgd
                v = (torch.matmul(kxy, s) + dxkxy) / self.particles.shape[0]
                self.particles = self.particles + self.config.training.particle_lr * v

            if ep % self.config.training.checkpoint_frq == 0 or ep == (
                    self.config.training.n_iter - 1):
                # print(lbd)
                train_acc, train_llk = self.model.bce_llk(
                    self.particles, X_train, y_train)
                logging.info("train acc: {:.3f}, train llk: {:.4f}".format(
                    train_acc, train_llk))
                test_acc, test_llk = self.model.bce_llk(
                    self.particles, X_test, y_test)
                bound_eq = self.boundary_eq(self.particles)
                ratio_out = np.sum(bound_eq.detach().cpu().numpy() >
                                   0) / self.config.training.n_particle
                target_loss = bound_eq.mean() + self.config.model.target_ub
                target_loss_test = self.boundary_eq_test(
                    self.particles).mean() + self.config.model.target_ub
                logging.info(
                    "Ep: {}, test acc: {:.3f}, test llk: {:.4f}, target loss: {:.4f}, ratio_out: {:.4f}, test target loss: {:.4f}"
                    .format(ep, test_acc, test_llk, target_loss, ratio_out,
                            target_loss_test))
                info["test_acc"] += [test_acc]
                info["test_llk"] += [test_llk]
                info["target_loss_test"] += [target_loss_test]
                info["target_loss"] += [target_loss]
                info[" train_llk"] += [train_llk]
                info["ratio_out"] += [ratio_out]

                torch.save(self.particles.detach().cpu(),
                           os.path.join(self.path, "checkpoint.pth"))
                torch.save(info,
                           os.path.join(self.path, "info.pth"))


class ControlledSVGD(RUNNER):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = config.optim.alpha
        self.kernel = Gaussian_kernel(h=config.optim.h_kernel)

    def laplace_boundary_eq(self, Z):
        def sample_rademacher_like(y):
            return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1
        dup_Z = Z.view(Z.shape[0], -1)
        dup_Z.requires_grad_(True)
        nabla_bound_eq = self.nabla_boundary_eq(dup_Z)
        approx_tr_dzdx = torch.zeros(dup_Z.shape[0]).to(self.device)

        num_e = 5
        for _ in range(num_e):
            e = sample_rademacher_like(dup_Z)
            e_dzdx = torch.autograd.grad(nabla_bound_eq,
                                         dup_Z,
                                         e,
                                         create_graph=True)[0]
            e_dzdx_e = e_dzdx * e
            approx_tr_dzdx += e_dzdx_e.view(
                dup_Z.shape[0], -1).sum(dim=1) / num_e
        return approx_tr_dzdx

    def train(self):
        X_train, y_train, X_test, y_test = self.get_datasets()
        self.scale_sto = X_train.shape[0] / self.config.training.batch_size
        self.particles = self.init_weights(X_train, y_train)
        lbd = 1.0
        n_particle = self.particles.shape[0]

        info = {'test_acc': [], 'test_llk': [], 'target_loss_test': [],
                'target_loss': [], ' train_llk': [], "ratio_out": []}

        for ep in range(self.config.training.n_iter):
            N0 = X_train.shape[0]
            batch = [
                i % N0
                for i in range(ep * self.config.training.batch_size, (ep + 1) *
                               self.config.training.batch_size)
            ]
            x = X_train[batch]
            y = y_train[batch]
            score_target = self.model.score(self.particles, x, y,
                                            self.scale_sto)
            bound_eq = self.boundary_eq(self.particles)
            nabla_bound_eq = self.nabla_boundary_eq(self.particles)

            kxx = self.kernel.kxy(self.particles, self.particles)
            dkxx_sum_former, dkxx_sum_latter = self.kernel.dkxy_sum(
                self.particles, self.particles)

            # num = self.alpha * bound_eq.mean() + (nabla_bound_eq * (
            #     score_target * (kxx.sum(-1)[:, None]) + dkxx_sum_latter
            # )).sum()/self.particles.shape[0]

            kxx_0 = kxx - torch.diag_embed(kxx.diag())
            num = self.alpha * bound_eq.mean() + ((nabla_bound_eq.matmul(score_target.T)) * kxx_0).sum() / \
                n_particle / (n_particle-1) + torch.sum(nabla_bound_eq *
                                                        dkxx_sum_former) / n_particle / (n_particle-1)

            # den = torch.sum(nabla_bound_eq.matmul(nabla_bound_eq.T)
            #                 * kxx)/self.particles.shape[0]

            den = torch.sum(nabla_bound_eq.matmul(nabla_bound_eq.T)
                            * kxx_0) / n_particle / (n_particle-1)

            # update the multiplier lambda
            lbd = min(max(num / (den + 1e-8), torch.tensor(0.0)),
                      torch.tensor(1000.0))

            # update the particle
            with torch.no_grad():
                particles_grad = (kxx.matmul(score_target - lbd * nabla_bound_eq)
                                  + dkxx_sum_former) / n_particle
                self.particles = self.particles + self.config.training.particle_lr * particles_grad

            if ep % self.config.training.checkpoint_frq == 0 or ep == (
                    self.config.training.n_iter - 1):
                test_acc, test_llk = self.model.bce_llk(
                    self.particles, X_test, y_test)
                bound_eq = self.boundary_eq(self.particles)
                ratio_out = np.sum(bound_eq.detach().cpu().numpy() >
                                   0) / self.config.training.n_particle
                target_loss = bound_eq.mean() + self.config.model.target_ub
                target_loss_test = self.boundary_eq_test(
                    self.particles).mean() + self.config.model.target_ub
                logging.info(
                    "Ep: {}, test acc: {:.3f}, test llk: {:.4f}, target loss: {:.4f}, ratio_out: {:.4f}, test target loss: {:.4f}, lambda: {:.4f}"
                    .format(ep, test_acc, test_llk, target_loss, ratio_out,
                            target_loss_test, lbd))
                info["test_acc"] += [test_acc]
                info["test_llk"] += [test_llk]
                info["target_loss_test"] += [target_loss_test]
                info["target_loss"] += [target_loss]
                # info[" train_llk"] += [train_llk]
                info["ratio_out"] += [ratio_out]

                torch.save(info,
                           os.path.join(self.path, "info.pth"))
                torch.save(self.particles.detach().cpu(),
                           os.path.join(self.path, "checkpoint.pth"))


class MIED(RUNNER):

    def __init__(self, config):
        super().__init__(config)
        self.alpha = config.optim.alpha_db
        self.alpha_mied = config.optim.alpha_mied
        self.eps = config.optim.eps
        self.diag_mul = config.optim.diag_mul
        self.weight_dim = (config.data.input_dim + 1) * config.model.n_hidden + (
            config.model.n_hidden + 1) * config.data.output_dim + 1
        self.riesz_s = 2 * self.alpha_mied * self.weight_dim + 1e-4
        self.get_datasets()
        self.particles = self.init_weights(self.X_train, self.y_train)
        self.particles.requires_grad_(True)
        self.optimizer = torch.optim.Adam(
            [self.particles], lr=config.training.particle_lr
        )
        self.projector = DynamicBarrier(self.alpha, config.optim.max_proj_iter)

        self.load_path = config.load_path

    def compute_energy(self, Z, batchdataset, batchlabel, scale_sto):
        '''
        :param Z: (B, D)
        :return: a scalar, the weighted riesz energy
        '''

        log_p = self.model.logp(Z, batchdataset, batchlabel, scale_sto)  # (B,)

        B = Z.shape[0]
        diff = Z.unsqueeze(1) - Z.unsqueeze(0)  # (B, B, D)
        diff_norm_sqr = diff.square().sum(-1)  # (B, B)

        vals, _ = torch.topk(diff_norm_sqr, 2, dim=-1, largest=False)
        vals = vals.detach()[:, 1]

        # Use \phi(h_i / (1.3d)^{1/d}) for the diagonal term.
        vals = vals / math.pow(self.diag_mul *
                               self.weight_dim, 2.0 / self.weight_dim)
        diff_norm_sqr = diff_norm_sqr + torch.diag(vals)

        log_dist_sqr = (diff_norm_sqr + self.eps).log()  # (B, B)
        tmp = log_dist_sqr * -self.riesz_s / 2

        tmp2 = (log_p.unsqueeze(1) + log_p.unsqueeze(0))  # (B, B)
        tmp2 = tmp2 * -self.alpha  # (B, B)

        tmp = tmp + tmp2

        mask = torch.eye(B, device=Z.device, dtype=torch.bool)  # (B, B)
        mask = torch.logical_not(mask)  # (B, B)
        mask = torch.logical_or(mask,
                                torch.eye(B, device=Z.device,
                                          dtype=torch.bool))
        mask = mask.reshape(-1)
        tmp = tmp.reshape(-1)
        tmp = torch.masked_select(tmp, mask)

        energy = torch.logsumexp(tmp, 0)  # scalar

        return energy

    def train(self):
        X_train, y_train, X_test, y_test = self.X_train, self.y_train, self.X_test, self.y_test
        self.scale_sto = X_train.shape[0] / self.config.training.batch_size

        if self.load_path != "":
            state_dict = torch.load(os.path.join(
                self.load_path, "checkpoint.pth"))
            self.particles = state_dict["particle"].to(self.device)

        info = {'test_acc': [], 'test_llk': [], 'target_loss_test': [],
                'target_loss': [], 'train_llk': [], "ratio_out": []}

        for ep in range(self.config.training.n_iter):
            N0 = X_train.shape[0]
            batch = [
                i % N0
                for i in range(ep * self.config.training.batch_size, (ep + 1) *
                               self.config.training.batch_size)
            ]
            x = X_train[batch]
            y = y_train[batch]
            self.optimizer.zero_grad()
            F = self.compute_energy(
                self.particles, x, y, self.scale_sto)
            F.backward()
            update = -self.particles.grad.detach()
            boundary_eq = self.boundary_eq(self.particles)
            nabla_boundary_eq = self.nabla_boundary_eq(self.particles)
            # The projector may modify update.
            with torch.no_grad():
                update = self.projector.step(
                    update, boundary_eq.detach().unsqueeze(1), nabla_boundary_eq.detach().unsqueeze(1))
            self.particles.grad = -update.detach().clone()
            self.optimizer.step()

            if ep % self.config.training.checkpoint_frq == 0 or ep == (
                    self.config.training.n_iter - 1):
                with torch.no_grad():
                    #     logging.info("f_net: {:.4f}, z_net: {:.4f}".format(
                    #         torch.abs(self.f_net(self.particles)).mean(),
                    #         torch.abs(self.z_net(self.particles)).mean()))
                    # logging.info(
                    #     "Ep: {}, loss: {:.4f}, loss_edge: {:.4f}, ratio: {:.4f} weight :{:.4f}"
                    #     .format(ep, loss, loss_edge, loss_edge / loss, weight))
                    train_acc, train_llk = self.model.bce_llk(
                        self.particles, X_train, y_train)
                    logging.info("train acc: {:.3f}, train llk: {:.4f}".format(
                        train_acc, train_llk))
                    test_acc, test_llk = self.model.bce_llk(
                        self.particles, X_test, y_test)
                    bound_eq = self.boundary_eq(self.particles)
                    ratio_out = np.sum(bound_eq.detach().cpu().numpy() >
                                    0) / self.config.training.n_particle
                    target_loss = bound_eq.mean() + self.config.model.target_ub
                    target_loss_test = self.boundary_eq_test(
                        self.particles).mean() + self.config.model.target_ub
                    logging.info(
                        "Ep: {}, test acc: {:.3f}, test llk: {:.4f}, target loss: {:.4f}, ratio_out: {:.4f}, test target loss: {:.4f}"
                        .format(ep, test_acc, test_llk, target_loss, ratio_out,
                                target_loss_test))
                    state_dict = {
                        "particle": self.particles.detach().cpu(),
                        "optimizer": self.optimizer.state_dict()
                    }
                    info["test_acc"] += [test_acc]
                    info["test_llk"] += [test_llk]
                    info["target_loss_test"] += [target_loss_test]
                    info["target_loss"] += [target_loss]
                    info["train_llk"] += [train_llk]
                    info["ratio_out"] += [ratio_out]

                    torch.save(info,
                            os.path.join(self.path, "info.pth"))
                    torch.save(state_dict,
                            os.path.join(self.path, "checkpoint.pth"))
        return info
