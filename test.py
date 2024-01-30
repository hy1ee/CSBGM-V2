# test checkpoint pth

import torch
import matplotlib.pyplot as plt

from mnist_vae.src.model_def import VAE
from argparse import ArgumentParser 

def main(hparams):
    z = torch.randn(1, hparams.n_z)
    dir = './mnist_vae/checkPoint/model_best.pth' # best model
    checkpoint = torch.load(dir)

    model = VAE(hparams)
    model.load_state_dict(checkpoint['state_dict'])
    print(z)
    print(model)
    random_res = model.decoder(z).view(-1, 1, 28, 28)

    image_array = random_res.squeeze().detach().numpy()

    # 使用 Matplotlib 显示图像
    plt.imshow(image_array, cmap='gray')  # 假设是灰度图像
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Base parameters in VAE
    PARSER.add_argument('--n_input', type=int, default='784', help='The shape of MNIST tensor')
    PARSER.add_argument('--n_z', type=int, default='20', help='The shape of latent variable z')
    PARSER.add_argument('--h_dim', type=int, default='500', help='The dim of hidden')
    PARSER.add_argument('--batch_size', type=int, default='128', help='The dim of hidden')

    HPARAMS = PARSER.parse_args()
    main(HPARAMS)


