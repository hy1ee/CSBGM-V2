from argparse import ArgumentParser 
import numpy as np
import os
import utils
import mnist_vae.src.utils as mnist_utils
import mnist_estimators
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
def main(hparams):
    # A measurement_type
    mnist_utils.print_hparams(hparams)

    mnist_train, mnist_test = mnist_utils.mnist_dataloader(hparams)
    l2_losses, measurement_losses = [], []


    x_batch, _ = next(iter(mnist_train))
    print(x_batch.shape)

    def show_images_in_one_row(images1, images2, images3):
        fig, axes = plt.subplots(3, len(images1), figsize=(20, 8))
        for i, img in enumerate(images1):
            axes[0, i].imshow(np.squeeze(img), cmap='binary')
            axes[0, i].axis('off')

        for i, img in enumerate(images2):
            axes[1, i].imshow(np.squeeze(img), cmap='binary')
            axes[1, i].axis('off')

        for i, img in enumerate(images3):
            axes[2, i].imshow(np.squeeze(img), cmap='binary')
            axes[2, i].axis('off')
        plt.show()

    x_batch_numpy = x_batch.squeeze().numpy()
    # show_images_in_one_row(x_batch_numpy)

    x_batch = x_batch.view(hparams.batch_size, hparams.n_input)  # Size: torch.size([batch_size, n_input])
        # print(x_batch_val.shape)

    
    A = np.random.randn(hparams.n_input, hparams.num_measurements)  # 参考mnist_reconstruction.sh , 应该设定成一个列表
    A = torch.tensor(A, dtype=torch.float32)
    noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, hparams.num_measurements)
    y_batch = np.matmul(x_batch, A) + noise_batch  # measurement_type在vae上设定为gaussian

    x_batch_hat = mnist_estimators.vae_estimator(A, y_batch, hparams)
    x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28) # hparams.batch_size, 1, 

    x_batch_hat_bayes = mnist_estimators.vae_bayesian_estimator(A, y_batch, hparams)
    x_batch_hat_bayes = x_batch_hat_bayes.reshape(hparams.batch_size, 1, 28, 28) # hparams.batch_size, 1, 


    # print(f"x_batch_hat{x_batch_hat.shape} x_batch{x_batch_numpy.shape}")
    # x_batch_hat_numpy = x_batch_hat.squeeze().numpy()
    show_images_in_one_row(x_batch_numpy, x_batch_hat, x_batch_hat_bayes)


    # plt.figure(figsize=(8, 4))

    # # 将第一个数组显示在左侧
    # plt.subplot(1, 2, 1)
    # plt.imshow(x_batch_numpy, cmap='binary')
    # plt.title('Image 1')
    # plt.axis('off')

    # # 将第二个数组显示在右侧
    # plt.subplot(1, 2, 2)
    # plt.imshow(x_batch_hat, cmap='binary')
    # plt.title('Image 2')
    # plt.axis('off')

    # # 显示图像
    # plt.show()








    assert 1 == 2
    step = 0
    for batch_idx, (x_batch, _) in enumerate(mnist_train):
        step += 1
        x_batch = x_batch.view(hparams.batch_size, hparams.n_input)  # Size: torch.size([batch_size, n_input])
        # print(x_batch_val.shape)


        A = np.random.randn(hparams.n_input, hparams.num_measurements)  # 参考mnist_reconstruction.sh , 应该设定成一个列表
        A = torch.tensor(A, dtype=torch.float32)
        noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, hparams.num_measurements)
        y_batch = np.matmul(x_batch, A) + noise_batch  # measurement_type在vae上设定为gaussian

        # Create estimator
        # for model_type in hparams.model_types:
        x_batch_hat = mnist_estimators.vae_bayesian_estimator(A, y_batch, hparams)

        # x_batch_hat = mnist_estimators.vae_estimator(A, y_batch, hparams)

        # print(f"x_batch.shape:{x_batch.shape} | x_batch_hat:{x_batch_hat.shape} | A.shape:{A.shape} | y_batch.shape:{y_batch.shape}")
        
        
        # print(x_batch_hat)
        # print(x_batch)
        # print(y_batch)
        # temp = x_hat[6, :]
        # image1 = temp.reshape(28, 28)

        # temp = x_batch[6, :]
        # image2 = temp.reshape(28, 28)
        # fig, axs = plt.subplots(1, 2, figsize=(18, 12))
        # axs[0].imshow(image1, cmap='gray')
        # axs[0].set_title('x_hat')

        # axs[1].imshow(image2, cmap='gray')
        # axs[1].set_title('x_batch')
        # plt.tight_layout()
        # plt.show()

        # print(x_hat.shape)
        # print(x_batch.shape)
        # assert 1 == 2



        l2_loss = utils.get_l2_loss(x_batch_hat, x_batch)
        measurement_loss = utils.get_measurement_loss(x_batch_hat, A, y_batch)
        l2_losses.append(l2_loss)
        measurement_losses.append(measurement_loss)

        print(l2_loss)
        print(measurement_loss)

                # visualize reconstructed result at the beginning of each epoch
        # x_concat = torch.cat([x_batch_hat[0].view(-1, 1, 28, 28), x_batch[0].view(-1, 1, 28, 28)], dim=3)
        # save_image(x_concat, './%s/reconstructed-%d.png')


        if step % 100 == 0:
            print(l2_loss)
            print(measurement_loss)
        if step >= 500:
            break
    print(np.mean(l2_losses))

if __name__ == '__main__':
    PARSER = ArgumentParser()

    # input
    PARSER.add_argument('--batch_size', type=int, default=10, help='How many examples are processed together')
    PARSER.add_argument('--n_input', type=int, default=784, help='How many examples are processed together')

    PARSER.add_argument('--num-measurements', type=int, default=400, help='number of gaussian measurements')
    PARSER.add_argument('--noise_std', type=int, default=0.1, help='number of gaussian measurements')

    # VAE model
    PARSER.add_argument('--n_z', type=int, default='20', help='The shape of latent variable z')
    PARSER.add_argument('--h_dim', type=int, default='500', help='The dim of hidden')

    PARSER.add_argument('--mloss1_weight', type=float, default=0.0, help='L1 measurement loss weight')
    PARSER.add_argument('--mloss2_weight', type=float, default=1.0, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior_weight', type=float, default=0.1, help='weight on z prior')

    PARSER.add_argument('--num_random_restarts', type=float, default=5, help='weight on z prior')
    PARSER.add_argument('--max_update_iter', type=float, default=1000, help='weight on z prior')

    PARSER.add_argument('--mnist_dir', type=str, default='./mnist_vae/checkPoint/model_best.pth', help='weight on z prior')

    # bayesian inference
    PARSER.add_argument('--theta_loss_weight', type=float, default=0.1, help='weight on z prior')

    HPARAMS = PARSER.parse_args()

    main(HPARAMS)
    


