from torch import nn

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