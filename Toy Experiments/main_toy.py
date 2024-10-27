import argparse
import logging
import os
import torch
import torch.optim as optim
from target_models_2d import Ring, Cardioid, DoubleMoon
import matplotlib.pyplot as plt
from models.models_2d import F_net, G_net
from utils.annealing import annealing

targets = {"Ring": Ring, "Cardioid": Cardioid, "DoubleMoon": DoubleMoon}
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets_name', default='Ring', type=str)
    parser.add_argument('--sigma', default= 1, type=float)
    parser.add_argument('--r_min', default= 1, type=float)
    parser.add_argument('--r_max', default= 2, type=float)
    parser.add_argument('--date', default= "2024-05-21", type=str)
    parser.add_argument('--f_latent_dim', default= 256, type=int)
    parser.add_argument('--f_lr', default= 0.005, type=float)
    parser.add_argument('--p_item', default= 2, type=float)
    parser.add_argument('--num_particle', default= 5000, type=int)
    parser.add_argument('--n_epoch', default= 10000, type=int)
    parser.add_argument('--f_iter', default= 3, type=int)
    parser.add_argument('--master_stepsize', default= 0.01, type=float)
    parser.add_argument('--edge_width', default= 0.1, type=float)
    parser.add_argument('--annealing', action='store_true')
    parser.add_argument('--requires_znet', action='store_true')
    parser.add_argument('--requiers_bound_loss', action='store_true')

    args = parser.parse_args()
    return args

class ConstrainedGWG(object):
    def __init__(self, args=None):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.targets_name == "Ring":
            self.model = Ring(self.device, sigma=self.args.sigma, r_min=self.args.r_min, r_max=self.args.r_max)
        elif self.args.targets_name == "Cardioid": 
            self.model = Cardioid(self.device, sigma=1,shape_param=1.2)
        elif self.args.targets_name == "DoubleMoon": 
            self.model = DoubleMoon(self.device)
            self.target_bd = -2.0
        self.particles = torch.randn([self.args.num_particle, 2]).to(self.device)
        self.f_net = F_net(2,self.args.f_latent_dim).to(self.device)
        self.z_net = G_net(2,self.args.f_latent_dim).to(self.device)
        self.requires_znet = args.requires_znet
        self.requiers_bound_loss = args.requiers_bound_loss
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
        activ_in = torch.ones(x.shape[0],1).to(self.device) 
        activ_in[ self.model.bondary_eq(x) > 0] = 0 
        nabla_bound_eq = self.model.nabla_bound(x) * 1
        active_edge = torch.zeros(x.shape[0],1).to(self.device) 
        edge_x_index = self.model.bondary_eq(
                    x - torch.sign(self.model.bondary_eq(x)[:,None])*self.args.edge_width * self.model.nabla_bound(x)/(self.model.nabla_bound(x).norm(2,dim=-1)[:,None])
                    ) * self.model.bondary_eq(x) < 0
        active_edge[edge_x_index] = 1
        if self.requires_znet:
            return (self.f_net(x) * activ_in 
                    - (1-activ_in) * nabla_bound_eq/(nabla_bound_eq.norm(2,dim=-1)[:,None]+0.0001)
                    - (self.z_net(x)**2) * nabla_bound_eq * activ_in # inner
                    )
        else:
            return (self.f_net(x) * activ_in 
                    - (1-activ_in) * nabla_bound_eq/(nabla_bound_eq.norm(2,dim=-1)[:,None]+0.0001)
                    ) 
    def train(self):
        # get model
        f_optim = optim.Adam(self.f_net.parameters(), lr=self.args.f_lr)
        z_optim = optim.Adam(self.z_net.parameters(), lr=self.args.f_lr)
        scheduler_f = torch.optim.lr_scheduler.StepLR(f_optim, step_size=3000, gamma=0.9)
        scheduler_z = torch.optim.lr_scheduler.StepLR(z_optim, step_size=3000, gamma=0.9)
        if self.args.targets_name == "DoubleMoon":
            self.model.bd = self.target_bd
        annealing_coef = lambda t: annealing(t, warm_up_interval = self.args.n_epoch//2, anneal = self.args.annealing)
        for ep in range(self.args.n_epoch):
            score_target = self.model.score(self.particles) * annealing_coef(ep)
            self.edge_sample_ind = self.model.bondary_eq(
                    self.particles - torch.sign(self.model.bondary_eq(self.particles)[:,None])*self.args.edge_width * self.model.nabla_bound(self.particles)/(self.model.nabla_bound(self.particles).norm(2,dim=-1)[:,None])
                    ) * self.model.bondary_eq(self.particles) < 0
            edge_sample = self.particles[self.edge_sample_ind]
            if ep % 100 == 0 or ep == (self.args.n_epoch-1):
                fig, ax = plt.subplots(figsize=(5, 5))
                self.model.contour_plot(ax, fnet = self.F_constrained, samples = [self.particles,edge_sample], save_to_path=self.args.save_to_path.replace("final.pt", str(ep)+".png"),quiver=True)
                plt.close()
            self.args.edge_width = max(self.args.edge_width/1.0002,0.05)
            for i in range(self.args.f_iter):                   
                self.particles.requires_grad_(True)
                f_value = self.F_constrained(self.particles)
                weight = (edge_sample.shape[0])/self.particles.shape[0] * 1/self.args.edge_width
                
                f_optim.zero_grad()
                z_optim.zero_grad()
                if self.requiers_bound_loss:
                    loss_edge = weight * torch.sum(self.F_constrained(edge_sample) * \
                                                    self.model.nabla_bound(edge_sample)/self.model.nabla_bound(edge_sample).norm(2,dim=-1)[:,None])\
                                                        /edge_sample.shape[0] if weight > 0 else 0
                else:
                    loss_edge = 0.0
                loss = (-torch.sum(score_target * f_value) - torch.sum(self.divergence_approx(f_value, self.particles)) + torch.norm(f_value, p=self.args.p_item)**self.args.p_item /self.args.p_item)/f_value.shape[0] \
                    + loss_edge
                loss.backward()

                f_optim.step()
                z_optim.step()
                scheduler_f.step()   
                scheduler_z.step()            
                self.particles.requires_grad_(False)
            
            # update the particle
            with torch.no_grad():
                gdgrad = self.F_constrained(self.particles)
                self.particles = self.particles + self.args.master_stepsize * gdgrad
            if ep % 100 == 0 or ep == (self.args.n_epoch-1):
                with torch.no_grad():
                    logging.info("f_net: {:.4f}, z_net: {:.4f}".format(torch.abs(self.f_net(self.particles)).mean(),torch.abs(self.z_net(self.particles)).mean()))
                logging.info("Ep: {}, loss: {:.4f}, loss_edge: {:.4f}, ratio: {:.4f} weight :{:.4f}".format(ep, loss, loss_edge, loss_edge/loss,weight))



if __name__ == '__main__':
    
    args = get_args()
    folder = os.path.join('results_b_{}_z_{}'.format(args.requiers_bound_loss, args.requires_znet), args.targets_name+'-'+args.date)
    os.makedirs(folder, exist_ok=True)
    args.save_to_path = os.path.join(folder, 'final.pt')
    args.logpath = os.path.join(folder, 'final.log')
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(args.logpath)
    formatter = logging.Formatter('')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(logging.INFO)
    logger.info('### Training Settings ###')
    for name, value in vars(args).items():
        logger.info('{} : {}'.format(name, value))
    args.logger = logger

    run_ConstrainedGWG = ConstrainedGWG(args)
    run_ConstrainedGWG.train()