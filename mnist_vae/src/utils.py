import os
import shutil

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

def mnist_dataloader(hparams):
    transform = transforms.Compose([transforms.ToTensor()])     # , transforms.Normalize((0.5,), (0.5,))

    # Download MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_train = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    mnist_test = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=True)

    return mnist_train, mnist_test

def print_hparams(hparams):
    print('')
    for temp in dir(hparams):
        if temp[:1] != '_':
            print('{0} = {1}'.format(temp, getattr(hparams, temp)))
    print('')


def get_loss(x_hat, x, z_mean, z_log_sigma_sq):
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum') # BCE
    KLD = 0.5 * torch.sum(torch.exp(z_log_sigma_sq) + torch.pow(z_mean, 2) - 1. - z_log_sigma_sq) # KL-divergence
    loss = BCE + KLD
    return loss, BCE, KLD

def save_checkpoint(state, is_best, outdir):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')  
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)  
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)
