import os
import shutil

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F

def mnist_dataloader(hparams):
    transform = transforms.Compose([transforms.ToTensor()])     # , transforms.Normalize((0.5,), (0.5,))

    # Download MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_train = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    mnist_test = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=True)

    return mnist_train, mnist_test