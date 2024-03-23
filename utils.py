import numpy as np
import torch
import matplotlib.pyplot as plt 
import torch.nn.functional as F

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


def print_hparams(hparams):
    print('')
    for temp in dir(hparams):
        if temp[:1] != '_':
            print('{0} = {1}'.format(temp, getattr(hparams, temp)))
    print('')


def convert_type(x):
    if isinstance(x, torch.Tensor):
        return x.detach().numpy()
    else:
        return x


def get_l2_loss(x_hat, x):
    """Get L2 loss between the two images"""
    x, x_hat = convert_type(x), convert_type(x_hat)
    assert x_hat.shape == x.shape
    return np.mean((x_hat - x)**2)


def get_measurement_loss(x_hat, A, y):
    """Get measurement loss of the estimated image"""

    x_hat, y, A = convert_type(x_hat), convert_type(y), convert_type(A)

    if A is None:
        y_hat = x_hat
    else:
        y_hat = np.matmul(x_hat, A)
    assert y_hat.shape == y.shape
    return np.mean((y - y_hat) ** 2)



def show_images(images_list, labels, hparams):
    loss_save_path = './LossResult'
    plt.figure(figsize=(18, 12))
    if hparams.batch_size == 1:
        n = len(images_list)
        fig, axes = plt.subplots(n, 1, figsize=(8, 20))

        for idx, (img, label) in enumerate(zip(images_list, labels)):
            axes[idx].imshow(np.squeeze(img), cmap='binary')
            axes[idx].axis('off')
            # axes[idx].set_title(label)
            axes[idx].text(-0.3, 0.5, f"{label}", fontsize=16, va='center', ha='right', transform=axes[idx].transAxes)

        plt.savefig('./%s/cs_image_bs1.png' % (loss_save_path))
        plt.show()
    
    else:
        n = len(images_list)
        fig, axes = plt.subplots(n, len(images_list[0]), figsize = (20, 8))

        for idx, (img_list, label) in enumerate(zip(images_list, labels)):
            for i, img in enumerate(img_list):
                
                axes[idx, i].imshow(np.squeeze(img), cmap='binary', vmin=np.min(img), vmax=np.max(img)) # , vmin=np.min(img), vmax=np.max(img)
                axes[idx, i].axis('off')
                
            axes[idx, 0].text(-0.3, 0.5, f"{label}", fontsize=16, va='center', ha='right', transform=axes[idx, 0].transAxes)

        plt.savefig('./%s/cs_image_bs10.png' % (loss_save_path))
        plt.show()
        

def mse_between_elements(lst):
    first_element = lst[0]
    mse_values = []
    for _, element in enumerate(lst[1:], start=1):
        if first_element.shape != element.shape:
            element = np.squeeze(element, axis=1)
        mse = np.mean((element - first_element) ** 2)
        mse_values.append(mse)
    return mse_values



def ssim_loss(y_batch, y_hat_batch, window_size=11, size_average=True):
        # Padding for the window
        padding = window_size // 2
        
        # Compute mean values
        mu_x = F.avg_pool1d(y_batch, window_size, stride=1, padding=padding)
        mu_y = F.avg_pool1d(y_hat_batch, window_size, stride=1, padding=padding)
        
        # Compute variances and covariances
        sigma_x_sq = F.avg_pool1d(y_batch**2, window_size, stride=1, padding=padding) - mu_x**2
        sigma_y_sq = F.avg_pool1d(y_hat_batch**2, window_size, stride=1, padding=padding) - mu_y**2
        sigma_xy = F.avg_pool1d(y_batch * y_hat_batch, window_size, stride=1, padding=padding) - mu_x * mu_y
        
        # Constants
        C1 = (0.01)**2
        C2 = (0.03)**2
        
        # SSIM formula
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        if size_average:
            return torch.mean((1 - ssim_map) / 2)
        else:
            return torch.mean((1 - ssim_map) / 2, dim=1)
        

def get_z(hparams, z_dim):
    if hparams.get_z_method == 'Gaussian':
        z = np.random.normal(0,hparams.temperature,[hparams.batch_size, z_dim])
        z = torch.tensor(z).float()
        z.requires_grad_(True)
    if hparams.get_z_method == 'Fixed':
        z = torch.ones([hparams.batch_size, z_dim], dtype=torch.float32, requires_grad=True)

    return z
