import os
import numpy as np
import utils
import torch
from torch import optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
# from model_def import VAE
from argparse import ArgumentParser 
from model_def import VAE_NVP
# Set device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def main(hparams):
    # Set up some stuff according to hparams
    utils.print_hparams(hparams)
    mnist_train, mnist_test = utils.mnist_dataloader(hparams)

    # Model
    model = VAE_NVP(hparams).to(device)
    optimizer = optim.Adam(model.parameters(), lr = hparams.learning_rate)

    # Load Checkpoint
    start_epoch = 0

    if not os.path.exists(hparams.result_dir):
        os.makedirs(hparams.result_dir)


    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    # Training
    Total_losses = []
    for epoch in range(start_epoch, hparams.training_epochs):
        loss_batch = []

        for batch_idx, (x_batch_val, _) in enumerate(mnist_train):
            x_batch_val = x_batch_val.to(device)

            # Training forward
            x_hat, z_mean, z_log_sigma_sq, likelihood = model(x_batch_val)
            nll_loss = torch.mean(likelihood)
            loss, BCE, KLD = utils.get_loss(x_hat, x_batch_val, z_mean, z_log_sigma_sq)

            loss += nll_loss
            loss_batch.append(loss.item())

            # Training Backward
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()

            # print statistics every 100 batch
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{hparams.training_epochs}], Batch [{batch_idx + 1}/{len(mnist_train.dataset)/hparams.batch_size}]: \
                      Total-loss = {loss.item()/hparams.batch_size}, BCE-Loss = {BCE.item()/hparams.batch_size}, \
                        KLD-loss = {KLD.item()/hparams.batch_size},  nll-loss = {nll_loss.item()/hparams.batch_size}")

            if batch_idx == 0:
                # visualize reconstructed result at the beginning of each epoch
                x_concat = torch.cat([x_batch_val.view(-1, 1, 28, 28), x_hat.view(-1, 1, 28, 28)], dim=3)
                save_image(x_concat, './%s/reconstructed-%d.png' % (hparams.result_dir, epoch + 1))

        Total_losses.append(np.sum(loss_batch) / len(mnist_train.dataset))  


        if (epoch+1) % 5 == 0:
            save_path = os.path.join(hparams.save_dir, '{}.pt'.format(epoch+1))
            torch.save(model.state_dict(), save_path)

            z_random_sample = torch.randn(hparams.batch_size, hparams.n_z)# .to(device)  
            z_random_sample = z_random_sample.to(device)  

            # decoder
            random_res = model.decoder(z_random_sample)

            save_image(random_res.view(-1, 1, 28, 28), './%s/random_sampled-%d.png' % (hparams.result_dir, epoch + 1))

    x_plt = range(len(Total_losses))

    plt.plot(x_plt, Total_losses, label='Total Loss')

    plt.title('VAE_RealNVP Loss')
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./%s/loss.png' % (hparams.result_dir))



if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Base parameters in VAE
    PARSER.add_argument('--n_input', type=int, default='784', help='The shape of MNIST tensor')
    PARSER.add_argument('--n_z', type=int, default='20', help='The shape of latent variable z')
    PARSER.add_argument('--vae_h_dim', type=int, default='500', help='The dim of hidden')
    PARSER.add_argument('--batch_size', type=int, default='128', help='The dim of hidden')


    # Coupling Layer Parameters
    PARSER.add_argument('--vae_nvp_h_dim', type=int, default='200', help='Size of the hidden layers in each coupling layer in RealNVP')
    PARSER.add_argument('--flow_n', type=int, default='9', help='Number of affine coupling layers in RealNVP')


    # Training Parameters
    PARSER.add_argument('--learning_rate', type=float, default='1e-3', help='The learning rate of optimize')
    PARSER.add_argument('--training_epochs', type=int, default='50', help='The training epochs of optimize')

    # Testing Parameters
    PARSER.add_argument('--test_every', type=int, default='10', help='Test after every epochs')

    # Checkpoint dir
    PARSER.add_argument('--result_dir', type=str, default='./mnist_vae_nvp/VaeNvpResult', help='Output directory')
    PARSER.add_argument('--save_dir', type=str, default='./mnist_vae_nvp/checkPoint', help='Model saving directory')
    PARSER.add_argument('--use_cuda', type=bool, default=True, help='Device')
    HPARAMS = PARSER.parse_args()
    main(HPARAMS)