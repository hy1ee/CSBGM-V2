import os
import shutil

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F

def mnist_dataloader(hparams):
    transform = transforms.Compose([transforms.ToTensor()])

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
    """
    每训练一定的epochs后， 判断损失函数是否是目前最优的，并保存模型的参数
    :param state: 需要保存的参数，数据类型为dict
    :param is_best: 说明是否为目前最优的
    :param outdir: 保存文件夹
    :return:
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')  # join函数创建子文件夹，也就是把第二个参数对应的文件保存在'outdir'里
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)  # 把state保存在checkpoint_file文件夹中
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)
