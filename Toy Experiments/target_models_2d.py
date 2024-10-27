import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class Ring(object):
    def __init__(self, device="cpu", sigma = 2, r_min=1, r_max=2, edge_width = 0.1):
        self.device = device
        self.sigma = sigma
        self.r_min = r_min
        self.r_max = r_max
        self.edge_width = edge_width
    def bondary_eq(self,X):
        return 1/4 * (torch.sum(X**2,dim=-1) - self.r_min**2) * (torch.sum(X**2,dim=-1) - self.r_max**2)
    def nabla_bound(self,X):
        return X * ((X**2).sum(dim=-1)[:,None] -(self.r_max**2 + self.r_min**2)/2)
    def logp(self, X):
        logp_ori = -0.5 * 2 * np.log(2 * np.pi) - 2 * np.log(self.sigma) -1/(2*self.sigma**2) * torch.sum(X**2,dim=-1)
        logp_ori[torch.logical_or(self.r_min>X.norm(2,dim=1),X.norm(2,dim=1) > self.r_max)] = -1000
        return logp_ori
    def sample(self,bs = 1000):
        samples = torch.randn([bs * 10,2]).to(self.device) * self.sigma
        return (samples[torch.logical_and(self.r_min<samples.norm(2,dim=1),samples.norm(2,dim=1) < self.r_max)])[:bs]
    def score(self, X):
        score_ori = -X/self.sigma**2
        score_ori[X.norm(2,dim=1) > self.r_max] = 0
        score_ori[X.norm(2,dim=1) < self.r_min] = 0
        return score_ori
    def contour_plot(self, ax, fnet=None, samples=None, save_to_path="./result.png", fig_title = "", quiver=False, num_pt = 5000, plot_edge=True):
        bbox = [-3, 3, -3, 3]
        xx, yy = np.mgrid[bbox[0]:bbox[1]:500j, bbox[2]:bbox[3]:500j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(-np.reshape(self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(), xx.shape))
        if samples is None:
            samples = self.sample(num_pt)
            edge_sample = samples[self.bondary_eq(
                    samples - np.sign(self.bondary_eq(samples)[:,None])*self.edge_width * self.nabla_bound(samples)/(self.nabla_bound(samples).norm(2,dim=-1)[:,None])
                    ) * self.bondary_eq(samples) < 0]
            samples = samples.cpu().numpy()
            edge_sample = edge_sample.cpu().numpy()
            samples = [samples, edge_sample]
        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
        ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels = 11)
        ax.plot(samples[0].cpu()[:, 0], samples[0].cpu()[:,1], '.', markersize= 2, color='#ff7f0e')
        if plot_edge:
            ax.plot(samples[1].cpu()[:, 0], samples[1].cpu()[:,1], '.', markersize= 2, color='red')
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            if fnet is None:
                scores = np.reshape(self.nabla_bound(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(), cpositions.T.shape)
            else:
                scores = np.reshape(fnet(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(), cpositions.T.shape)
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        ax.set_title(fig_title, fontsize = 30, y=1.04)
        if save_to_path is not None:
            torch.save(scores,save_to_path.replace(".png","scores.pt"))
            torch.save(samples,save_to_path.replace(".png",".pt"))
            plt.savefig(save_to_path, bbox_inches='tight')


class Cardioid(object):
    def __init__(self, device="cpu", sigma = 2, shape_param = 1.2, edge_width = 0.1):
        self.device = device
        self.sigma = sigma
        self.shape_param = shape_param
        self.edge_width = edge_width
    def bondary_eq(self,X):
        return X[:,0]**2 + (X[:,1] * self.shape_param - torch.pow(X[:,0]**2,1/3))**2 - 4
    def nabla_bound(self,X):
        auxi_term = (X[:,1] * self.shape_param - torch.pow(X[:,0]**2+0.0001,1/3))

        return torch.cat(((2 * X[:,0] - 4/3 * (torch.sigmoid(X[:,0]) -0.5) * 2 * 1/(torch.pow(torch.abs(X[:,0])+0.0001,1/3)+0.0001) * auxi_term)[:,None], 
                          (self.shape_param * auxi_term)[:,None]
                          )
                          ,dim=-1
                        )
    def logp(self, X):
        bond_eq = self.bondary_eq(X)
        logp_ori = -0.5 * 2 * np.log(2 * np.pi) - 2 * np.log(self.sigma) -1/(2*self.sigma**2) * torch.sum(X**2,dim=-1)
        logp_ori[bond_eq > 0] = -1000
        return logp_ori
    def sample(self,bs = 1000):
        X = torch.randn([bs * 10,2]).to(self.device) * self.sigma
        bond_eq = self.bondary_eq(X)
        return (X[(bond_eq) < 0])[:bs]
    def score(self, X):
        score_ori = -X/self.sigma**2
        return score_ori
    def contour_plot(self, ax, fnet=None, samples=None, save_to_path="./result.png", fig_title = "", quiver=False, num_pt = 5000, plot_edge=True):
        bbox = [-3, 3, -3, 3]
        xx, yy = np.mgrid[bbox[0]:bbox[1]:500j, bbox[2]:bbox[3]:500j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(-np.reshape(self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(), xx.shape))
        if samples is None:
            samples = self.sample(num_pt)
            edge_sample = samples[self.bondary_eq(
                    samples - np.sign(self.bondary_eq(samples)[:,None])*self.edge_width * self.nabla_bound(samples)/(self.nabla_bound(samples).norm(2,dim=-1)[:,None])
                    ) * self.bondary_eq(samples) < 0]
            samples = samples.cpu().numpy()
            edge_sample = edge_sample.cpu().numpy()
            samples = [samples, edge_sample]
        
        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
        ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels = 11)
        ax.plot(samples[0].cpu()[:, 0], samples[0].cpu()[:,1], '.', markersize= 2, color='#ff7f0e')
        if plot_edge:
            ax.plot(samples[1].cpu()[:, 0], samples[1].cpu()[:,1], '.', markersize= 2, color='red')
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            if fnet is None:
                scores = np.reshape(self.nabla_bound(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(), cpositions.T.shape)
            else:
                scores = np.reshape(fnet(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(), cpositions.T.shape)
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        ax.set_title(fig_title, fontsize = 30, y=1.04)
        if save_to_path is not None:
            torch.save(scores,save_to_path.replace(".png","scores.pt"))
            torch.save(samples,save_to_path.replace(".png",".pt"))
            plt.savefig(save_to_path, bbox_inches='tight')


class DoubleMoon(object):
    def __init__(self, device="cpu", bd = -5.0, edge_width = 0.1):
        self.device = device
        self.bd = bd
        self.edge_width = edge_width
    def bondary_eq(self, X):
        means_d1 = torch.tensor([[3.0, -3.0]]).to(self.device)
        logp_ori = (-2 * (torch.sqrt(torch.sum(X**2, dim=-1)) - 3.0)**2 +
                     torch.logsumexp(-2 * ((X[:,0])[:,None]- means_d1)**2, dim = 1)
                    )
        
        return -logp_ori + self.bd
    def nabla_bound(self,X):
        score_part_1 = -4 * (X - (3.0 * (X+1e-6)/(torch.sqrt(torch.sum(X**2, dim = -1))[:,None] + 1e-6)))
        X1_minus_means = (X[:,0])[:,None]- torch.tensor([[3.0, -3.0]]).to(X.device)
        score_part_2 = -4 * (X1_minus_means * F.softmax((-2) * X1_minus_means**2, dim=-1)).sum(dim=-1)
        nabla_bound_X = score_part_1
        nabla_bound_X[:,0] = nabla_bound_X[:,0] + score_part_2
        return - nabla_bound_X
    def logp(self, X):
        means_d1 = torch.tensor([[3.0, -3.0]]).to(self.device)
        logp_ori = (-2 * (torch.sqrt(torch.sum(X**2, dim=-1)) - 3.0)**2 + torch.logsumexp(-2 * ((X[:,0])[:,None]- means_d1)**2, dim = 1))
        logp_ori[logp_ori < self.bd] = -10000
        return logp_ori
    def score(self, X):
        score_part_1 = -4 * (X - (3.0 * (X+1e-6)/(torch.sqrt(torch.sum(X**2, dim = -1))[:,None] + 1e-6)))
        X1_minus_means = (X[:,0])[:,None]- torch.tensor([[3.0, -3.0]]).to(X.device)
        score_part_2 = -4 * (X1_minus_means * F.softmax((-2) * X1_minus_means**2, dim=-1)).sum(dim=-1)
        score_ori = score_part_1
        score_ori[:,0] = score_ori[:,0] + score_part_2
        score_ori[self.bondary_eq(X) > 0] = 0
        return score_ori
    def sample(self, bs=1000, loop = 10000, epsilon_0 = 5 * 1e-4, alpha = 0, accept_rate=False):
        """
        In general, we can sample the agent ground truth with langevin dynamics
        """
        Z = torch.zeros(bs * 10, 2).to(self.device)
        for t in range(0, loop):
            compu_targetscore = self.score(Z)
            learn_rate = epsilon_0/(1+t)**alpha
            Z = Z + learn_rate/2 * compu_targetscore + np.sqrt(learn_rate) * torch.randn([Z.shape[0],2]).to(self.device)
        bond_eq = self.bondary_eq(Z)
        constrained_samples = Z[(bond_eq) < 0]
        accept_rate = constrained_samples.shape[0]/Z.shape[0]
        if accept_rate: 
            return constrained_samples[:bs], accept_rate
        else:
            return constrained_samples[:bs]
    
    def contour_plot(self, ax, fnet=None, samples=None, save_to_path="./result.png", fig_title = "", quiver=False, num_pt = 5000, plot_edge=True):
        bbox = [-4, 4, -4, 4]
        xx, yy = np.mgrid[bbox[0]:bbox[1]:500j, bbox[2]:bbox[3]:500j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(-np.reshape(self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(), xx.shape))
        if samples is None:
            samples, accept_rate = self.sample(num_pt,accept_rate=True)
            assert samples.shape[0] == num_pt, "The number of samples does not meet the requirements."
            edge_sample = samples[self.bondary_eq(
                    samples - np.sign(self.bondary_eq(samples)[:,None])*self.edge_width * self.nabla_bound(samples)/(self.nabla_bound(samples).norm(2,dim=-1)[:,None])
                    ) * self.bondary_eq(samples) < 0]
            samples = samples.cpu().numpy()
            edge_sample = edge_sample.cpu().numpy()
            samples = [samples, edge_sample]
        
        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
        ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels = 11)
        ax.plot(samples[0].cpu()[:, 0], samples[0].cpu()[:,1], '.', markersize= 2, color='#ff7f0e')
        if plot_edge:
            ax.plot(samples[1].cpu()[:, 0], samples[1].cpu()[:,1], '.', markersize= 2, color='red')
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            if fnet is None:
                scores = np.reshape(self.nabla_bound(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(), cpositions.T.shape)
            else:
                scores = np.reshape(fnet(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(), cpositions.T.shape)
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        ax.set_title(fig_title, fontsize = 30, y=1.04)
        if save_to_path is not None:
            torch.save(scores,save_to_path.replace(".png","scores.pt"))
            torch.save(samples,save_to_path.replace(".png",".pt"))
            plt.savefig(save_to_path, bbox_inches='tight')


class Block_mirror_hard(object):
    def __init__(self, device="cpu", sigma=1, shape_param=2, edge_width=0.1,mode_mu=-2):
        self.device = device
        self.sigma = sigma
        self.shape_param = shape_param
        self.edge_width = edge_width
        self.mode_mu=mode_mu

    def safe_log(self,X):
        return torch.log(torch.max(torch.tensor(1e-12),X))

    def soft_sigin(self, X):
        # return (torch.sigmoid(X) - 0.5) * 2
        lam = 10
        return ((1 + torch.exp(-lam * X)) ** (-1) - 0.5) * 2

    def bondary_eq(self, X):
        return (torch.abs(X[:, 0] - X[:, 1]) + torch.abs(X[:, 0] + X[:, 1])) - self.shape_param ** 2

    # def nabla_bound(self,X):
    #     return torch.cat(((torch.sign(X[:,0]+X[:,1]) + torch.sign(X[:,0]-X[:,1]))[:,None],
    #                       (torch.sign(X[:,0]+X[:,1]) - torch.sign(X[:,0]-X[:,1]))[:,None]
    #                       )
    #                       ,dim=-1
    #                     )
    def nabla_bound(self, X):
        return torch.cat(((self.soft_sigin(X[:, 0] + X[:, 1]) + self.soft_sigin(X[:, 0] - X[:, 1]))[:, None],
                          (self.soft_sigin(X[:, 0] + X[:, 1]) - self.soft_sigin(X[:, 0] - X[:, 1]))[:, None]
                          )
                         , dim=-1
                         )

    def logp(self, X):
        bond_eq = self.bondary_eq(X)
        logp_ori = -0.5 * 2 * np.log(2 * np.pi)+np.log(16) -0.5* (torch.log(2+X[:, 0])-torch.log(2-X[:, 0])-self.mode_mu)**2 -torch.log(2+X[:, 0])-torch.log(2-X[:, 0]) -0.5* (torch.log(2+X[:, 1])-torch.log(2-X[:, 1])-self.mode_mu)**2 -torch.log(2+X[:, 1])-torch.log(2-X[:, 1])
        logp_ori[bond_eq > 0] = -1000
        return logp_ori

    # def sample(self, bs=10000):
    def sample(self, bs=100):
        X = torch.randn([bs * 10, 2]).to(self.device) * self.sigma + torch.tensor(
            [[self.mode_mu, self.mode_mu]], device=self.device)
        X=2-4/(torch.exp(X)+1)
        return X

    def score(self, X):
        score_ori = -(self.safe_log(2+X)-self.safe_log(2-X)-self.mode_mu)*4/(torch.exp(self.safe_log((2+X)*(2-X))))+2*X/(torch.exp(self.safe_log((2+X)*(2-X))))
        score_ori[self.bondary_eq(X) > 0] = 0
        return score_ori

    def contour_plot(self, ax, fnet=None, samples=None, save_to_path="./result.png", fig_title="", quiver=False,
                     num_pt=5000, plot_edge=True):
        bbox = [-3, 3, -3, 3]
        xx, yy = np.mgrid[bbox[0]:bbox[1]:500j, bbox[2]:bbox[3]:500j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(-np.reshape(self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(), xx.shape))
        if samples is None:
            samples = self.sample(num_pt)
            edge_sample = samples[self.bondary_eq(
                samples - np.sign(self.bondary_eq(samples)[:, None]) * self.edge_width * self.nabla_bound(samples) / (
                self.nabla_bound(samples).norm(2, dim=-1)[:, None])
            ) * self.bondary_eq(samples) < 0]
            samples = samples.cpu().numpy()
            edge_sample = edge_sample.cpu().numpy()
            samples = [samples, edge_sample]

        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
        ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels=11)
        ax.plot(samples[0].cpu()[:, 0], samples[0].cpu()[:, 1], '.', markersize=2, color='#ff7f0e')
        if plot_edge:
            ax.plot(samples[1].cpu()[:, 0], samples[1].cpu()[:, 1], '.', markersize=2, color='red')
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            if fnet is None:
                scores = np.reshape(self.nabla_bound(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(),
                                    cpositions.T.shape)
            else:
                scores = np.reshape(fnet(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(),
                                    cpositions.T.shape)
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_title(fig_title, fontsize=30, y=1.04)
        if save_to_path is not None:
            torch.save(scores, save_to_path.replace(".png", "scores.pt"))
            torch.save(samples, save_to_path.replace(".png", ".pt"))
            plt.savefig(save_to_path, bbox_inches='tight')


class Block_mirror_harder(object):
    def __init__(self, device="cpu", sigma=1, shape_param=2, edge_width=0.1,mode_mu=-1000):
        self.device = device
        self.sigma = sigma
        self.shape_param = shape_param
        self.edge_width = edge_width
        self.mode_mu=mode_mu

    def safe_log(self,X):
        return torch.log(torch.max(torch.tensor(1e-12),X))

    def soft_sigin(self, X):
        # return (torch.sigmoid(X) - 0.5) * 2
        lam = 10
        return ((1 + torch.exp(-lam * X)) ** (-1) - 0.5) * 2

    def bondary_eq(self, X):
        return (torch.abs(X[:, 0] - X[:, 1]) + torch.abs(X[:, 0] + X[:, 1])) - self.shape_param ** 2

    # def nabla_bound(self,X):
    #     return torch.cat(((torch.sign(X[:,0]+X[:,1]) + torch.sign(X[:,0]-X[:,1]))[:,None],
    #                       (torch.sign(X[:,0]+X[:,1]) - torch.sign(X[:,0]-X[:,1]))[:,None]
    #                       )
    #                       ,dim=-1
    #                     )
    def nabla_bound(self, X):
        return torch.cat(((self.soft_sigin(X[:, 0] + X[:, 1]) + self.soft_sigin(X[:, 0] - X[:, 1]))[:, None],
                          (self.soft_sigin(X[:, 0] + X[:, 1]) - self.soft_sigin(X[:, 0] - X[:, 1]))[:, None]
                          )
                         , dim=-1
                         )

    def logp(self, X):
        bond_eq = self.bondary_eq(X)
        logp_ori = -0.5 * 2 * np.log(2 * np.pi)+np.log(16) -0.5* (torch.log(2+X[:, 0])-torch.log(2-X[:, 0])-self.mode_mu)**2 -torch.log(2+X[:, 0])-torch.log(2-X[:, 0]) -0.5* (torch.log(2+X[:, 1])-torch.log(2-X[:, 1]))**2 -torch.log(2+X[:, 1])-torch.log(2-X[:, 1])
        logp_ori[bond_eq > 0] = -1000
        return logp_ori

    # def sample(self, bs=10000):
    def sample(self, bs=100):
        X = torch.randn([bs * 10, 2]).to(self.device) * self.sigma + torch.tensor(
            [[self.mode_mu, 0]], device=self.device)
        X=2-4/(torch.exp(X)+1)
        return X

    def score(self, X):

        score_ori = torch.cat((torch.unsqueeze(-(self.safe_log(2+X[:, 0])-self.safe_log(2-X[:, 0])-self.mode_mu)*4/(torch.exp(self.safe_log((2+X[:, 0])*(2-X[:, 0]))))+2*X[:, 0]/(torch.exp(self.safe_log((2+X[:, 0])*(2-X[:, 0])))),1),torch.unsqueeze(-(self.safe_log(2+X[:, 1])-self.safe_log(2-X[:, 1]))*4/(torch.exp(self.safe_log((2+X[:, 1])*(2-X[:, 1]))))+2*X[:, 1]/(torch.exp(self.safe_log((2+X[:, 1])*(2-X[:, 1])))),1)),1)
        score_ori[self.bondary_eq(X) > 0] = 0
        return score_ori

    def contour_plot(self, ax, fnet=None, samples=None, save_to_path="./result.png", fig_title="", quiver=False,
                     num_pt=5000, plot_edge=True):
        bbox = [-3, 3, -3, 3]
        xx, yy = np.mgrid[bbox[0]:bbox[1]:500j, bbox[2]:bbox[3]:500j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(-np.reshape(self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(), xx.shape))
        if samples is None:
            samples = self.sample(num_pt)
            edge_sample = samples[self.bondary_eq(
                samples - np.sign(self.bondary_eq(samples)[:, None]) * self.edge_width * self.nabla_bound(samples) / (
                self.nabla_bound(samples).norm(2, dim=-1)[:, None])
            ) * self.bondary_eq(samples) < 0]
            samples = samples.cpu().numpy()
            edge_sample = edge_sample.cpu().numpy()
            samples = [samples, edge_sample]

        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
        ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels=11)
        ax.plot(samples[0].cpu()[:, 0], samples[0].cpu()[:, 1], '.', markersize=2, color='#ff7f0e')
        if plot_edge:
            ax.plot(samples[1].cpu()[:, 0], samples[1].cpu()[:, 1], '.', markersize=2, color='red')
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            if fnet is None:
                scores = np.reshape(self.nabla_bound(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(),
                                    cpositions.T.shape)
            else:
                scores = np.reshape(fnet(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(),
                                    cpositions.T.shape)
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_title(fig_title, fontsize=30, y=1.04)
        if save_to_path is not None:
            torch.save(scores, save_to_path.replace(".png", "scores.pt"))
            torch.save(samples, save_to_path.replace(".png", ".pt"))
            plt.savefig(save_to_path, bbox_inches='tight')


class Block_nineMG(object):
    def __init__(self, device="cpu", sigma=1, shape_param=2, edge_width=0.1):
        self.device = device
        self.sigma = sigma
        self.shape_param = shape_param
        self.edge_width = edge_width

    def soft_sigin(self, X):
        return (torch.sigmoid(X) - 0.5) * 2

    def bondary_eq(self, X):
        return (torch.abs(X[:, 0] - X[:, 1]) + torch.abs(X[:, 0] + X[:, 1])) - self.shape_param ** 2

    def nabla_bound(self, X):
        return torch.cat(((self.soft_sigin(X[:, 0] + X[:, 1]) + self.soft_sigin(X[:, 0] - X[:, 1]))[:, None],
                          (self.soft_sigin(X[:, 0] + X[:, 1]) - self.soft_sigin(X[:, 0] - X[:, 1]))[:, None]
                          )
                         , dim=-1
                         )

    def logp(self, X):
        means = torch.tensor([[-1, -1], [0, -1],[1, -1],[-1, -1],[0, -1],[1, -1],[-1, -1],[0, -1],[1, -1]]).to(self.device)
        bond_eq = self.bondary_eq(X)
        logp_ori = -0.5 * 2 * np.log(2 * np.pi) - np.log(9.0) + torch.logsumexp(
            -torch.sum((X.unsqueeze(1) - means.unsqueeze(0)) ** 2, dim=-1) / 2. / (0.2) ** 2
            , dim=1)
        logp_ori[bond_eq > 0] = -1000
        return logp_ori

    def score(self, X):

        import torch.distributions as D
        K = 9
        torch.manual_seed(1)
        mix = D.Categorical(torch.ones(K,).to(self.device))
        comp = D.Independent(D.Normal(
                     torch.tensor([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]]).to(self.device), torch.ones(K,2).to(self.device)*0.2), 1)
        gmm = D.MixtureSameFamily(mix, comp)

        X = X.requires_grad_(True)

        log_prob = gmm.log_prob(X)
        score_ori = torch.autograd.grad(log_prob.sum(), X)[0].detach()
        score_ori[self.bondary_eq(X) > 0] = 0

        return score_ori





    def sample(self, bs=1000, loop=10000, epsilon_0=5 * 1e-4, alpha=0.2):
        """
        In general, we can sample the agent ground truth with langevin dynamics
        """
        # Z = torch.zeros(bs, 2).to(self.device)
        # for t in range(0, loop):
        #     compu_targetscore = self.score(Z)
        #     learn_rate = epsilon_0/(1+t)**alpha
        #     Z = Z + learn_rate/2 * compu_targetscore + np.sqrt(learn_rate) * torch.randn([Z.shape[0],2]).to(self.device)
        # return Z
        """
        Insteade, it is accessible for the ground truth in the special case.
        """
        import torch.distributions as D
        K = 9
        torch.manual_seed(1)
        mix = D.Categorical(torch.ones(K, ).to(self.device))
        comp = D.Independent(D.Normal(
            torch.tensor([[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]).to(
                self.device), torch.ones(K, 2).to(self.device) * 0.2), 1)
        gmm = D.MixtureSameFamily(mix, comp)

        # sample = gmm.sample((100,)).cpu()

        sample = gmm.sample((1000,))

        bond_eq = self.bondary_eq(sample)
        return sample[(bond_eq) < 0]

    def contour_plot(self, ax, fnet=None, samples=None, save_to_path="./result.png", fig_title="", quiver=False,
                     num_pt=5000, plot_edge=True):
        bbox = [-3, 3, -3, 3]
        xx, yy = np.mgrid[bbox[0]:bbox[1]:500j, bbox[2]:bbox[3]:500j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(-np.reshape(self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(), xx.shape))
        if samples is None:
            samples = self.sample(num_pt)
            edge_sample = samples[self.bondary_eq(
                samples - np.sign(self.bondary_eq(samples)[:, None]) * self.edge_width * self.nabla_bound(samples) / (
                self.nabla_bound(samples).norm(2, dim=-1)[:, None])
            ) * self.bondary_eq(samples) < 0]
            samples = samples.cpu().numpy()
            edge_sample = edge_sample.cpu().numpy()
            samples = [samples, edge_sample]

        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
        ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels=11)
        ax.plot(samples[0].cpu()[:, 0], samples[0].cpu()[:, 1], '.', markersize=2, color='#ff7f0e')
        if plot_edge:
            ax.plot(samples[1].cpu()[:, 0], samples[1].cpu()[:, 1], '.', markersize=2, color='red')
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            if fnet is None:
                scores = np.reshape(self.nabla_bound(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(),
                                    cpositions.T.shape)
            else:
                scores = np.reshape(fnet(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(),
                                    cpositions.T.shape)
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_title(fig_title, fontsize=30, y=1.04)
        if save_to_path is not None:
            torch.save(scores, save_to_path.replace(".png", "scores.pt"))
            torch.save(samples, save_to_path.replace(".png", ".pt"))
            plt.savefig(save_to_path, bbox_inches='tight')




class Block_nineMG_edge(object):
    def __init__(self, device="cpu", sigma=1, shape_param=2, edge_width=0.1):
        self.device = device
        self.sigma = sigma
        self.shape_param = shape_param
        self.edge_width = edge_width
        self.var = 0.2
        self.center = 1.7


    def soft_sigin(self, X):
        return (torch.sigmoid(X) - 0.5) * 2

    def bondary_eq(self, X):
        return (torch.abs(X[:, 0] - X[:, 1]) + torch.abs(X[:, 0] + X[:, 1])) - self.shape_param ** 2

    def nabla_bound(self, X):
        return torch.cat(((self.soft_sigin(X[:, 0] + X[:, 1]) + self.soft_sigin(X[:, 0] - X[:, 1]))[:, None],
                          (self.soft_sigin(X[:, 0] + X[:, 1]) - self.soft_sigin(X[:, 0] - X[:, 1]))[:, None]
                          )
                         , dim=-1
                         )

    def logp(self, X):
        means = torch.tensor([[-self.center, -self.center], [0, -self.center],[self.center, -self.center],[-self.center, 0],[0, 0],[self.center, 0],[-self.center, self.center],[0, self.center],[self.center, self.center]]).to(self.device)
        bond_eq = self.bondary_eq(X)
        logp_ori = -0.5 * 2 * np.log(2 * np.pi) - np.log(9.0) + torch.logsumexp(
            -torch.sum((X.unsqueeze(1) - means.unsqueeze(0)) ** 2, dim=-1) / 2. / (self.var) ** 2
            , dim=1)
        logp_ori[bond_eq > 0] = -1000
        return logp_ori



    def score(self, X):

        import torch.distributions as D
        K = 9
        torch.manual_seed(1)
        mix = D.Categorical(torch.ones(K,).to(self.device))
        comp = D.Independent(D.Normal(
                     torch.tensor([[-self.center,-self.center],[0,-self.center],[self.center,-self.center],[-self.center,0],[0,0],[self.center,0],[-self.center,self.center],[0,self.center],[self.center,self.center]]).to(self.device), torch.ones(K,2).to(self.device)*self.var), 1)
        gmm = D.MixtureSameFamily(mix, comp)

        X = X.requires_grad_(True)

        log_prob = gmm.log_prob(X)
        score_ori = torch.autograd.grad(log_prob.sum(), X)[0].detach()
        score_ori[self.bondary_eq(X) > 0] = 0

        return score_ori





    def sample(self, bs=1000, loop=10000, epsilon_0=5 * 1e-4, alpha=0.2):
        """
        In general, we can sample the agent ground truth with langevin dynamics
        """
        # Z = torch.zeros(bs, 2).to(self.device)
        # for t in range(0, loop):
        #     compu_targetscore = self.score(Z)
        #     learn_rate = epsilon_0/(1+t)**alpha
        #     Z = Z + learn_rate/2 * compu_targetscore + np.sqrt(learn_rate) * torch.randn([Z.shape[0],2]).to(self.device)
        # return Z
        """
        Insteade, it is accessible for the ground truth in the special case.
        """
        import torch.distributions as D
        K = 9
        torch.manual_seed(1)
        mix = D.Categorical(torch.ones(K, ).to(self.device))
        comp = D.Independent(D.Normal(
            torch.tensor([[-self.center, -self.center], [0, -self.center], [self.center, -self.center], [-self.center, 0], [0, 0], [self.center, 0], [-self.center, self.center], [0, self.center], [self.center, self.center]]).to(
                self.device), torch.ones(K, 2).to(self.device) * self.var), 1)
        gmm = D.MixtureSameFamily(mix, comp)

        # sample = gmm.sample((100,)).cpu()

        sample = gmm.sample((1000,))

        bond_eq = self.bondary_eq(sample)
        return sample[(bond_eq) < 0]

    def contour_plot(self, ax, fnet=None, samples=None, save_to_path="./result.png", fig_title="", quiver=False,
                     num_pt=5000, plot_edge=True):
        bbox = [-3, 3, -3, 3]
        xx, yy = np.mgrid[bbox[0]:bbox[1]:500j, bbox[2]:bbox[3]:500j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = -np.log(-np.reshape(self.logp(torch.Tensor(positions.T).to(self.device)).cpu().numpy(), xx.shape))
        if samples is None:
            samples = self.sample(num_pt)
            edge_sample = samples[self.bondary_eq(
                samples - np.sign(self.bondary_eq(samples)[:, None]) * self.edge_width * self.nabla_bound(samples) / (
                self.nabla_bound(samples).norm(2, dim=-1)[:, None])
            ) * self.bondary_eq(samples) < 0]
            samples = samples.cpu().numpy()
            edge_sample = edge_sample.cpu().numpy()
            samples = [samples, edge_sample]

        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
        ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels=11)
        ax.plot(samples[0].cpu()[:, 0], samples[0].cpu()[:, 1], '.', markersize=2, color='#ff7f0e')
        if plot_edge:
            ax.plot(samples[1].cpu()[:, 0], samples[1].cpu()[:, 1], '.', markersize=2, color='red')
        if quiver:
            cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
            if fnet is None:
                scores = np.reshape(self.nabla_bound(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(),
                                    cpositions.T.shape)
            else:
                scores = np.reshape(fnet(torch.Tensor(cpositions.T).to(self.device)).detach().cpu().numpy(),
                                    cpositions.T.shape)
            ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_title(fig_title, fontsize=30, y=1.04)
        if save_to_path is not None:
            torch.save(scores, save_to_path.replace(".png", "scores.pt"))
            torch.save(samples, save_to_path.replace(".png", ".pt"))
            plt.savefig(save_to_path, bbox_inches='tight')