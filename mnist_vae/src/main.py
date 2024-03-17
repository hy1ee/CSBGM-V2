import os
import numpy as np
import utils
import torch
from torch import optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from model_def import VAE
from argparse import ArgumentParser 

# Set device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def test(model, optimizer, mnist_test, epoch, best_test_loss, hparams):
    test_avg_loss = 0.0
    with torch.no_grad():  
        for test_batch_index, (test_x, _) in enumerate(mnist_test):
            test_x = test_x.to(device)
            test_x_hat, test_mu, test_log_var = model(test_x)
            test_loss, test_BCE, test_KLD = utils.get_loss(test_x_hat, test_x, test_mu, test_log_var)
            test_avg_loss += test_loss

        test_avg_loss /= len(mnist_test.dataset)

        # Random Sample

        z = torch.randn(hparams.batch_size, hparams.n_z).to(device)  
        random_res = model.decoder(z).view(-1, 1, 28, 28)

        save_image(random_res, './%s/random_sampled-%d.png' % (hparams.result_dir, epoch + 1))

        is_best = test_avg_loss < best_test_loss
        best_test_loss = min(test_avg_loss, best_test_loss)
        utils.save_checkpoint({
            'epoch': epoch, 
            'best_test_loss': best_test_loss,  
            'state_dict': model.state_dict(),  
            'optimizer': optimizer.state_dict(),
        }, is_best, hparams.save_dir)

        return best_test_loss



def main(hparams):
    # Set up some stuff according to hparams
    utils.print_hparams(hparams)
    mnist_train, mnist_test = utils.mnist_dataloader(hparams)

    # Check
    x, _ = iter(mnist_train).__next__()
    print(f"img shape:{x.shape}") # should be 'torch.Size([batch_size, 1, 28, 28])'

    # Model
    model = VAE(hparams).to(device)
    print(f"The structure of model is shown below: \n{model}")
    optimizer = optim.Adam(model.parameters(), lr = hparams.learning_rate)

    # Load Checkpoint
    start_epoch = 0
    best_test_loss = np.finfo('f').max
    if hparams.resume:
        if os.path.isfile(hparams.resume):
            print('=> loading checkpoint %s' % hparams.resume)
            checkpoint = torch.load(hparams.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint {hparams.resume}")
        else:
            print(f"=> no checkpoint found at {hparams.resume}")

    if not os.path.exists(hparams.result_dir):
        os.makedirs(hparams.result_dir)

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    # Training
    loss_epoch = []

    # save for plot
    Total_losses, BCE_losses, KLD_losses = [], [], []

    for epoch in range(start_epoch, hparams.training_epochs):
        loss_batch = []

        for batch_idx, (x_batch_val, _) in enumerate(mnist_train):
            x_batch_val = x_batch_val.to(device)

            # Training forward
            x_hat, z_mean, z_log_sigma_sq = model(x_batch_val)
            loss, BCE, KLD = utils.get_loss(x_hat, x_batch_val, z_mean, z_log_sigma_sq)
            loss_batch.append(loss.item())

            # Training Backward
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()

            # print statistics every 100 batch
            if (batch_idx + 1) % 100 == 0:

                Total_losses.append(loss.item()/hparams.batch_size)
                BCE_losses.append(BCE.item()/hparams.batch_size)
                KLD_losses.append(KLD.item()/hparams.batch_size)

                print(f"Epoch [{epoch+1}/{hparams.training_epochs}], Batch [{batch_idx + 1}/{len(mnist_train.dataset)/hparams.batch_size}]: \
                      Total-loss = {Total_losses[-1]}, BCE-Loss = {BCE_losses[-1]}, KLD-loss = {KLD_losses[-1]}")

            if batch_idx == 0:
                # visualize reconstructed result at the beginning of each epoch
                x_concat = torch.cat([x_batch_val.view(-1, 1, 28, 28), x_hat.view(-1, 1, 28, 28)], dim=3)
                save_image(x_concat, './%s/reconstructed-%d.png' % (hparams.result_dir, epoch + 1))

        loss_epoch.append(np.sum(loss_batch) / len(mnist_train.dataset))  

        if (epoch + 1) % hparams.test_every == 0:
            best_test_loss = test(model, optimizer, mnist_test, epoch, best_test_loss, hparams)

    # Plot losses
        
    x_plt = range(len(Total_losses))

    plt.plot(x_plt, Total_losses, label='Total Loss')
    plt.plot(x_plt, BCE_losses, label='BCE Loss')
    plt.plot(x_plt, KLD_losses, label='KLD Loss')

    plt.title('VAE Loss')
    plt.xlabel('Training Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./%s/loss_comparison.png' % (hparams.result_dir))


    return loss_epoch



if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Base parameters in VAE
    PARSER.add_argument('--n_input', type=int, default='784', help='The dim of input')
    PARSER.add_argument('--n_z', type=int, default='20', help='The dim of latent z')
    PARSER.add_argument('--vae_h_dim', type=int, default='500', help='The dim of hidden in VAE')
    PARSER.add_argument('--batch_size', type=int, default='100', help='Batch size for each epoch')

    # Training Parameters
    PARSER.add_argument('--learning_rate', type=float, default='1e-3', help='The learning rate of optimize')
    PARSER.add_argument('--training_epochs', type=int, default='100', help='The training epochs')

    # Testing Parameters
    PARSER.add_argument('--test_every', type=int, default='10', help='Test after every epochs')

    # Checkpoint dir
    PARSER.add_argument('--resume', type=str, default='None', help='Path to latest checkpoint')
    PARSER.add_argument('--save_dir', type=str, default='./mnist_vae/checkPoint', help='Path to store model parameters')
    PARSER.add_argument('--result_dir', type=str, default='./mnist_vae/VAEResult', help='Path to store model results')
    
    HPARAMS = PARSER.parse_args()
    main(HPARAMS)
