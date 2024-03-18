import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, hparams):
        super(VAE, self).__init__()

        self.n_input = hparams.n_input
        self.n_z = hparams.n_z
        self.h_dim = hparams.vae_h_dim

        # Encoder
        self.fc1 = nn.Linear(self.n_input, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.h_dim)
        self.fc3_mean = nn.Linear(self.h_dim, self.n_z)
        self.fc4_log_var = nn.Linear(self.h_dim, self.n_z)

        # Decoder
        self.fc5 = nn.Linear(self.n_z, self.h_dim)
        self.fc6 = nn.Linear(self.h_dim, self.h_dim)
        self.fc7 = nn.Linear(self.h_dim, self.n_input)

    def forward(self, x):
        """
        :param x: the input of our training model [b, batch_size, 1, 28, 28]
        :return: the result of our training model
        """
        batch_size = x.shape[0]  
        x = x.view(batch_size, self.n_input) 

        # encoder
        z_mean, z_log_sigma_sq = self.encoder(x)

        z_sampled = z_mean + torch.randn_like(torch.exp(z_log_sigma_sq * 0.5))
        # decoder
        x_hat = self.decoder(z_sampled)
        # reshape
        x_hat = x_hat.view(batch_size, 1, 28, 28)
        return x_hat, z_mean, z_log_sigma_sq

    def encoder(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        hidden1 = F.softplus(self.fc1(x))
        hidden2 = F.softplus(self.fc2(hidden1))
        z_mean = self.fc3_mean(hidden2)
        z_log_sigma_sq = self.fc4_log_var(hidden2)

        return z_mean, z_log_sigma_sq

    def decoder(self, z):
        """
        Given a sampled z, decoder/generator it back to image
        """
        hidden1 = F.softplus(self.fc5(z))
        hidden2 = F.softplus(self.fc6(hidden1))
        x_hat = torch.sigmoid(self.fc7(hidden2))  
        return x_hat


