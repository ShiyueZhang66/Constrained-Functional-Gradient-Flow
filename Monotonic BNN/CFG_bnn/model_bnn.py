import numpy as np
import torch
from torch import nn, autograd
from copy import deepcopy


class BNN(object):

    def __init__(self, device, d, a=1, b=0.1, n_hidden=50):
        '''
        d is the dimension of input.
        '''
        self.device = device
        self.a = a
        self.b = b
        self.n_hidden = n_hidden
        self.d = d
        self.dim_vars = (self.d + 1) * self.n_hidden + (self.n_hidden + 1) + 2
        self.dim_wb = self.dim_vars - 2

    def logp(self, Z, batchdataset, batchlabel, scale_sto=1, max_param=50.0):
        # Z is shaped with [1, self.dim_vars], assume that batchlabel is [n,1]
        """
        return the log posterior distribution P(W, log_gamma, log_lambda|Y,X), batchdataset, batchlabel normalize
        """
        log_gamma = Z[:, -2]
        log_lambda = Z[:, -1]
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(
            -1, self.d, self.n_hidden)  # [B, d, hidden]
        b1 = Z[:,
               (self.d) * self.n_hidden:(self.d + 1) * self.n_hidden].reshape(
                   -1, self.n_hidden)  # [B, hidden]
        W2 = Z[:, (self.d + 1) * self.n_hidden:(self.d + 1) * self.n_hidden +
               self.n_hidden][:, :, None]  # [B, hidden, 1]
        b2 = Z[:, -3].reshape(-1, 1)  # [B, 1]
        dnn_predict = (torch.matmul(
            torch.max(
                torch.matmul(batchdataset, W1) + b1[:, None, :],
                torch.tensor([0.0]).to(self.device)), W2) + b2[:, None, :]
                       )  # [B, n, 1]
        log_lik_data = -0.5 * batchdataset.shape[0] * (
            np.log(2 * np.pi) - log_gamma) - (gamma_ / 2) * torch.sum(
                ((dnn_predict - batchlabel).squeeze(2))**2, 1)
        log_prior_data = (self.a - 1) * log_gamma - self.b * gamma_ + log_gamma
        log_prior_w = -0.5 * self.dim_wb * (np.log(2 * np.pi) - log_lambda) - (
            lambda_ / 2) * ((W1**2).sum((1, 2)) + (W2**2).sum(
                (1, 2)) + (b1**2).sum(1) + (b2**2).sum(1)) + (
                    self.a - 1) * log_lambda - self.b * lambda_ + log_lambda
        return (log_lik_data * scale_sto + log_prior_data + log_prior_w)

    def score(self, Z, batchdataset, batchlabel, scale_sto=1, max_param=50.0):
        batch_Z = Z.shape[0]
        num_data = batchdataset.shape[0]

        log_gamma = Z[:, -2].reshape(-1, 1)  # [B, 1]
        log_lambda = Z[:, -1].reshape(-1, 1)
        gamma_ = torch.exp(log_gamma).clamp(max=max_param)
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(
            -1, self.d, self.n_hidden)  # [B, d, hidden]
        b1 = Z[:,
               (self.d) * self.n_hidden:(self.d + 1) * self.n_hidden].reshape(
                   -1, self.n_hidden)  # [B, hidden]
        W2 = Z[:, (self.d + 1) * self.n_hidden:(self.d + 1) * self.n_hidden +
               self.n_hidden][:, :, None]  # [B, hidden, 1]
        b2 = Z[:, -3].reshape(-1, 1)  # [B, 1]

        dnn_onelinear = torch.matmul(batchdataset, W1) + b1[:, None, :]
        dnn_relu_onelinear = torch.max(dnn_onelinear,
                                       torch.tensor([0.0]).to(self.device))
        dnn_grad_relu = (torch.sign(dnn_onelinear) +
                         1) / 2  # shape = [B, n, hidden]
        dnn_predict = (torch.matmul(dnn_relu_onelinear, W2) + b2[:, None, :]
                       )  # shape = [B,n,1]

        nabla_predict_b1 = dnn_grad_relu * W2.transpose(1, 2)  # [B, n, hidden]
        nabla_predict_W1 = nabla_predict_b1[:, :, None, :] * batchdataset[
            None, :, :,
            None]  # [B,n,d, hidden] # dnn_grad_relu[:,:,:,None] * batchdataset[None,:,None,:] * W2[:,:,:,None]
        nabla_predict_W2 = dnn_relu_onelinear  # [B,n, hidden]
        nabla_predict_b2 = torch.ones_like(dnn_predict).to(
            self.device)  # [B,n,1]

        nabla_predict_wb = torch.cat(
            (nabla_predict_W1.reshape(batch_Z, num_data, -1), nabla_predict_b1,
             nabla_predict_W2, nabla_predict_b2),
            dim=2)
        nabla_wb = scale_sto * gamma_ * (
            (batchlabel - dnn_predict) *
            nabla_predict_wb).sum(1) - lambda_ * Z[:, :-2]
        nabla_log_gamma = scale_sto * (
            0.5 * num_data - (gamma_ / 2) * torch.sum(
                (dnn_predict - batchlabel)**2, 1)) + (
                    self.a - 1) - self.b * gamma_ + 1  # [B, 1]
        nabla_log_lambda = 0.5 * self.dim_wb - lambda_ / 2 * (
            Z[:, :-2]**2).sum(1).unsqueeze(1) + (
                self.a - 1) - self.b * lambda_ + 1  # [B,1]
        return torch.cat((nabla_wb, nabla_log_gamma, nabla_log_lambda),
                         dim=1)  # shape = [B, self.dim_vars]

    def predict_y(self,
                  Z,
                  batchdataset,
                  mean_y_train,
                  std_y_train,
                  scale=True):
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(
            -1, self.d, self.n_hidden)  # [B, d, hidden]
        b1 = Z[:,
               (self.d) * self.n_hidden:(self.d + 1) * self.n_hidden].reshape(
                   -1, self.n_hidden)  # [B, hidden]
        W2 = Z[:, (self.d + 1) * self.n_hidden:(self.d + 1) * self.n_hidden +
               self.n_hidden][:, :, None]  # [B, hidden, 1]
        b2 = Z[:, -3].reshape(-1, 1)  # [B, 1]
        dnn_predict = (torch.matmul(
            torch.max(
                torch.matmul(batchdataset, W1) + b1[:, None, :],
                torch.tensor([0.0]).to(self.device)), W2) + b2[:, None, :])
        if scale:
            dnn_predict_true = dnn_predict * std_y_train + mean_y_train
            return dnn_predict_true
        else:
            return dnn_predict

    def mono_loss(self, Z, batchdataset, mean_y_train, std_y_train, index, ub):
        """
            index: an integer, index of monotone feature
            ub: upper bound of mono loss
        """
        # dup_data = deepcopy(batchdataset)
        # dup_data.requires_grad_(True)
        # y_hat = self.predict_y(Z,
        #                        dup_data,
        #                        mean_y_train,
        #                        std_y_train,
        #                        scale=False).squeeze(-1)  # [B, n]
        # y_hat.mean(1)

        y_hat = self.predict_y(Z,
                               batchdataset,
                               mean_y_train,
                               std_y_train,
                               scale=False).squeeze(-1)  # [B, n]
        eps = 1e-6
        batchdataset_right = deepcopy(batchdataset)
        batchdataset_right[:, index] = batchdataset_right[:, index] + eps
        y_hat_right = self.predict_y(Z,
                                     batchdataset_right,
                                     mean_y_train,
                                     std_y_train,
                                     scale=False).squeeze(-1)  # [B, n]
        mono_loss = torch.max(
            (y_hat - y_hat_right) / eps, torch.tensor(0.0)).mean(1) - ub  # [B]
        return mono_loss

    def nabla_mono_loss(self, Z, batchdataset, mean_y_train, std_y_train,
                        index, ub):
        dup_Z = Z.view(Z.shape[0], -1)
        dup_Z.requires_grad_(True)
        mono_loss = self.mono_loss(dup_Z, batchdataset, mean_y_train,
                                   std_y_train, index, ub).sum()
        Z_grad = autograd.grad(mono_loss,
                               dup_Z,
                               retain_graph=True,
                               create_graph=True)[0]
        return Z_grad


class F_net(nn.Module):

    def __init__(self, z_dim, latent_dim=128):
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


class Z_net(nn.Module):

    def __init__(self, z_dim, latent_dim=128):
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


class BNNClassifier(object):

    def __init__(self, device, d, a=1, b=0.1, n_hidden=50):
        '''
        d is the dimension of input.
        '''
        self.device = device
        self.a = a
        self.b = b
        self.n_hidden = n_hidden
        self.d = d
        self.dim_vars = (self.d + 1) * self.n_hidden + (self.n_hidden + 1) + 1
        self.dim_wb = self.dim_vars - 1

    def logp(self, Z, batchdataset, batchlabel, scale_sto=1, max_param=50.0):
        # Z is shaped with [1, self.dim_vars], assume that batchlabel is [n,1]
        """
        return the log posterior distribution P(W, log_gamma, log_lambda|Y,X), batchdataset, batchlabel normalize
        """
        log_lambda = Z[:, -1]
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(
            -1, self.d, self.n_hidden)  # [B, d, hidden]
        b1 = Z[:,
               (self.d) * self.n_hidden:(self.d + 1) * self.n_hidden].reshape(
                   -1, self.n_hidden)  # [B, hidden]
        W2 = Z[:, (self.d + 1) * self.n_hidden:(self.d + 1) * self.n_hidden +
               self.n_hidden][:, :, None]  # [B, hidden, 1]
        b2 = Z[:, -2].reshape(-1, 1)  # [B, 1]
        dnn_predict = (torch.matmul(
            torch.max(
                torch.matmul(batchdataset, W1) + b1[:, None, :],
                torch.tensor([0.0]).to(self.device)), W2) + b2[:, None, :]
                       )  # [B, n, 1]
        dnn_predict = dnn_predict.squeeze(2)
        log_lik_data = torch.matmul(
            dnn_predict, batchlabel).squeeze(1) - torch.logaddexp(
                torch.tensor([0.0]).to(self.device), dnn_predict).sum(1)
        log_prior_w = -0.5 * self.dim_wb * (np.log(2 * np.pi) - log_lambda) - (
            lambda_ / 2) * ((W1**2).sum((1, 2)) + (W2**2).sum(
                (1, 2)) + (b1**2).sum(1) + (b2**2).sum(1)) + (
                    self.a - 1) * log_lambda - self.b * lambda_ + log_lambda
        return log_lik_data * scale_sto + log_prior_w

    def score(self, Z, batchdataset, batchlabel, scale_sto=1, max_param=50.0):
        batch_Z = Z.shape[0]
        num_data = batchdataset.shape[0]

        log_lambda = Z[:, -1].reshape(-1, 1)
        lambda_ = torch.exp(log_lambda).clamp(max=max_param)
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(
            -1, self.d, self.n_hidden)  # [B, d, hidden]
        b1 = Z[:,
               (self.d) * self.n_hidden:(self.d + 1) * self.n_hidden].reshape(
                   -1, self.n_hidden)  # [B, hidden]
        W2 = Z[:, (self.d + 1) * self.n_hidden:(self.d + 1) * self.n_hidden +
               self.n_hidden][:, :, None]  # [B, hidden, 1]
        b2 = Z[:, -2].reshape(-1, 1)  # [B, 1]

        dnn_onelinear = torch.matmul(batchdataset, W1) + b1[:, None, :]
        dnn_relu_onelinear = torch.max(dnn_onelinear,
                                       torch.tensor([0.0]).to(self.device))
        dnn_grad_relu = (torch.sign(dnn_onelinear) +
                         1) / 2  # shape = [B, n, hidden]
        dnn_predict = (torch.matmul(dnn_relu_onelinear, W2) + b2[:, None, :]
                       )  # shape = [B,n,1]

        nabla_predict_b1 = dnn_grad_relu * W2.transpose(1, 2)  # [B, n, hidden]
        nabla_predict_W1 = nabla_predict_b1[:, :, None, :] * batchdataset[
            None, :, :,
            None]  # [B,n,d, hidden] # dnn_grad_relu[:,:,:,None] * batchdataset[None,:,None,:] * W2[:,:,:,None]
        nabla_predict_W2 = dnn_relu_onelinear  # [B,n, hidden]
        nabla_predict_b2 = torch.ones_like(dnn_predict).to(
            self.device)  # [B,n,1]

        nabla_predict_wb = torch.cat(
            (nabla_predict_W1.reshape(batch_Z, num_data, -1), nabla_predict_b1,
             nabla_predict_W2, nabla_predict_b2),
            dim=2)
        nabla_wb = scale_sto * ((batchlabel - torch.sigmoid(dnn_predict)) *
                                nabla_predict_wb).sum(1) - lambda_ * Z[:, :-1]

        nabla_log_lambda = 0.5 * self.dim_wb - lambda_ / 2 * (
            Z[:, :-1]**2).sum(1).unsqueeze(1) + (
                self.a - 1) - self.b * lambda_ + 1  # [B,1]
        return torch.cat((nabla_wb, nabla_log_lambda),
                         dim=1)  # shape = [B, self.dim_vars]

    def bce_llk(self, Z, batchdataset, batchlabel, max_param=50.0):
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(
            -1, self.d, self.n_hidden)  # [B, d, hidden]
        b1 = Z[:,
               (self.d) * self.n_hidden:(self.d + 1) * self.n_hidden].reshape(
                   -1, self.n_hidden)  # [B, hidden]
        W2 = Z[:, (self.d + 1) * self.n_hidden:(self.d + 1) * self.n_hidden +
               self.n_hidden][:, :, None]  # [B, hidden, 1]
        b2 = Z[:, -3].reshape(-1, 1)  # [B, 1]
        dnn_predict = (torch.matmul(
            torch.max(
                torch.matmul(batchdataset, W1) + b1[:, None, :],
                torch.tensor([0.0]).to(self.device)), W2) + b2[:, None, :])

        dnn_predict = dnn_predict.squeeze(2)
        y_hat = (torch.sigmoid(dnn_predict).log().mean(0) -
                 torch.tensor([0.5]).to(self.device).log()) > 0
        test_acc = torch.sum(
            y_hat == batchlabel.squeeze(-1)) / batchlabel.shape[0]

        test_llk = torch.matmul(
            dnn_predict, batchlabel).squeeze(1) - torch.logaddexp(
                torch.tensor([0.0]).to(self.device), dnn_predict).sum(1)
        test_llk = test_llk.mean(0) / batchlabel.shape[0]
        return test_acc.item(), test_llk.item()

    def predict_y(self, Z, batchdataset, logits=False):
        W1 = Z[:, :(self.d) * self.n_hidden].reshape(
            -1, self.d, self.n_hidden)  # [B, d, hidden]
        b1 = Z[:,
               (self.d) * self.n_hidden:(self.d + 1) * self.n_hidden].reshape(
                   -1, self.n_hidden)  # [B, hidden]
        W2 = Z[:, (self.d + 1) * self.n_hidden:(self.d + 1) * self.n_hidden +
               self.n_hidden][:, :, None]  # [B, hidden, 1]
        b2 = Z[:, -3].reshape(-1, 1)  # [B, 1]
        dnn_predict = (torch.matmul(
            torch.max(
                torch.matmul(batchdataset, W1) + b1[:, None, :],
                torch.tensor([0.0]).to(self.device)), W2) + b2[:, None, :])
        dnn_predict = dnn_predict.squeeze(2)
        if logits:
            return dnn_predict
        else:
            y_hat = (torch.sigmoid(dnn_predict).log().mean(0) -
                     torch.tensor([0.5]).to(self.device).log()) > 0
            return y_hat

    def mono_loss(self, Z, batchdataset, index, ub):
        """
            index: a list of intergers, indices of monotone feature
            ub: upper bound of mono loss
        """

        mono_loss = torch.tensor([0.0]).to(self.device)
        for i in index:
            eps = 1e-6
            batchdataset_right = deepcopy(batchdataset)
            batchdataset_right[:, i] = batchdataset_right[:, i] + eps / 2
            y_hat_right = self.predict_y(Z, batchdataset_right,
                                         logits=True).squeeze(-1)  # [B, n]

            batchdataset_left = deepcopy(batchdataset)
            batchdataset_left[:, i] = batchdataset_left[:, i] - eps / 2
            y_hat_left = self.predict_y(Z, batchdataset_left,
                                        logits=True).squeeze(-1)  # [B, n]
            mono_loss = mono_loss + torch.max((y_hat_left - y_hat_right) / eps,
                                              torch.tensor(0.0)).mean(1)  # [B]
        return mono_loss - ub

    def nabla_mono_loss(self, Z, batchdataset, index, ub):
        dup_Z = Z.view(Z.shape[0], -1)
        dup_Z.requires_grad_(True)
        mono_loss = self.mono_loss(dup_Z, batchdataset, index, ub).sum()
        Z_grad = autograd.grad(mono_loss,
                               dup_Z,
                               retain_graph=False,
                               create_graph=False)[0]
        return Z_grad

    def fair_loss(self, Z, batchdataset, batchattr, ub):
        """
            ub: upper bound of fair loss
        """

        y_hat = self.predict_y(Z, batchdataset,
                               logits=True).squeeze(-1)  # [B, N]
        batchattr = batchattr - batchattr.mean()
        batchattr = batchattr.repeat(y_hat.shape[0], 1)  # [B, N]

        fair_loss = torch.mean(y_hat * batchattr, dim=1)**2  # [B]
        return fair_loss - ub

    def nabla_fair_loss(self, Z, batchdataset, batchattr, ub):
        dup_Z = Z.view(Z.shape[0], -1)
        dup_Z.requires_grad_(True)
        fair_loss = self.fair_loss(dup_Z, batchdataset, batchattr, ub).sum()
        Z_grad = autograd.grad(fair_loss,
                               dup_Z,
                               retain_graph=False,
                               create_graph=False)[0]
        return Z_grad
