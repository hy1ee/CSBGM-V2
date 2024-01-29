import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

def model_input(hparams):
    """Create input tensors"""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if hparams.input_type == 'full_input':
        images = {i: image.numpy().flatten() for i, (image, _) in enumerate(mnist_test)}
        images = dict(list(images.items())[:hparams.num_input_images]) 

    return images


class Hparams(object):
    def __init__(self):
        self.n_hidden_recog_1 = 500  # 1st layer encoder neurons
        self.n_hidden_recog_2 = 500  # 2nd layer encoder neurons
        self.n_hidden_gener_1 = 500  # 1st layer decoder neurons
        self.n_hidden_gener_2 = 500  # 2nd layer decoder neurons
        self.n_input = 784           # MNIST data input (img shape: 28*28)
        self.n_z = 20                # dimensionality of latent space
        self.transfer_fct = nn.softplus


class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(hparams.n_input, hparams.n_hidden_recog_1)
        self.fc2 = nn.Linear(hparams.n_hidden_recog_1, hparams.n_hidden_recog_2)
        self.fc3_mean = nn.Linear(hparams.n_hidden_recog_2, hparams.n_z)
        self.fc4_log_var = nn.Linear(hparams.n_hidden_recog_2, hparams.n_z)

    def forward(self, x):
        hidden1 = F.softplus(self.fc1(x))
        hidden2 = F.softplus(self.fc2(hidden1))
        z_mean = self.fc3_mean(hidden2)
        z_log_var = self.fc4_log_var(hidden2)

        return z_mean, z_log_var
    

class Generator(nn.Module):
    def __init__(self, hparams):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(hparams.n_z, hparams.n_hidden_gener_1)
        self.fc2 = nn.Linear(hparams.n_hidden_gener_1, hparams.n_hidden_gener_2)
        self.fc3 = nn.Linear(hparams.n_hidden_gener_2, hparams.n_input)

    def forward(self, z):
        hidden1 = F.softplus(self.fc1(z))
        hidden2 = F.softplus(self.fc2(hidden1))
        logits = self.fc3(hidden2)
        x_reconstr_mean = torch.sigmoid(logits)

        return logits, x_reconstr_mean
    

def get_loss(x, logits, z_mean, z_log_sigma_sq):
    reconstr_losses = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')
    latent_losses = -0.5 * torch.sum(1 + z_log_sigma_sq - z_mean.pow(2) - torch.exp(z_log_sigma_sq))
    total_loss = (reconstr_losses + latent_losses) / x.size(0)  # Assuming x is a batch of samples
    return total_loss


def get_z_var(hparams, batch_size):
    z = torch.randn((batch_size, hparams.n_z), requires_grad=True)
    return z


def gen_restore_vars():
    restore_vars = ['gen/w1',
                    'gen/b1',
                    'gen/w2',
                    'gen/b2',
                    'gen/w3',
                    'gen/b3']
    return restore_vars