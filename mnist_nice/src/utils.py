import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def mnist_dataloader(hparams):
    transform = transforms.Compose([transforms.ToTensor()])     # , transforms.Normalize((0.5,), (0.5,))

    # Download MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_train = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    mnist_test = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=True)

    return mnist_train, mnist_test


class BestKeeper(object):
    """Class to keep the best stuff"""
    def __init__(self, hparams):
        self.batch_size = hparams.batch_size
        self.losses_val_best = [1e10 for _ in range(hparams.batch_size)]
        self.x_hat_batch_val_best = np.zeros((hparams.batch_size, hparams.n_input))

    def report(self, x_hat_batch_val, losses_val):
        for i in range(self.batch_size):
            if losses_val[i] < self.losses_val_best[i]:
                self.x_hat_batch_val_best[i, :] = x_hat_batch_val[i, :].detach().numpy()
                self.losses_val_best[i] = losses_val[i]

    def get_best(self):
        return self.x_hat_batch_val_best