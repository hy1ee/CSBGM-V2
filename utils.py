import numpy as np
import torch
import matplotlib.pyplot as plt 



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


def show_images(images_list, labels):
    """
    The list should include various estimation methods
    """

    n = len(images_list)
    fig, axes = plt.subplots(n, len(images_list[0]), figsize = (20, 8))

    for idx, (img_list, label) in enumerate(zip(images_list, labels)):
        for i, img in enumerate(img_list):

            axes[idx, i].imshow(np.squeeze(img), cmap='binary', vmin=np.min(img), vmax=np.max(img))
            axes[idx, i].axis('off')
            
        axes[idx, 0].text(-0.3, 0.5, f"{label}", fontsize=16, va='center', ha='right', transform=axes[idx, 0].transAxes)

    plt.show()