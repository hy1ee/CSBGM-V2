import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import torch.optim as optim
import torch
from torch import nn
import numpy as np
from torchvision import datasets, transforms
import itertools
import matplotlib.pyplot as plt
import utils
from train_ae import AutoEncoder
from argparse import ArgumentParser 
from torchvision.utils import save_image
from tqdm import tqdm
import random

import mnist_realnvp.src.real_nvp as real_nvp


def main(hparams):
    # Build the autoencoder
    if hparams.use_cuda:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = "cpu"
    autoencoder = AutoEncoder(hparams)
    autoencoder = autoencoder.to(device)
    autoencoder.load_state_dict(torch.load((hparams.AE_pretrained_dir)))

    mnist_train, mnist_test = utils.mnist_dataloader(hparams)

    embs = []
    for batch_idx, data in enumerate(mnist_train):

        with torch.no_grad():
            x, y = data

            x = x.to(device)

            _, emb = autoencoder(x)
            for i in range(len(emb)):
                embs.append((emb[i], y[i]))


    embs_loader = torch.utils.data.DataLoader(embs, hparams.batch_size)

    model = real_nvp.LinearRNVP(input_dim = hparams.n_input
                                , coupling_topology = [hparams.realnvp_h_dim]
                                , flow_n = hparams.flow_n
                                , batch_norm=True
                                , mask_type='odds'
                                , conditioning_size=None
                                , use_permutation=True
                                , single_function=True)
    
    model.train()
    opt = optim.Adam(itertools.chain(model.parameters()), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    if not os.path.exists(hparams.result_dir):
        os.makedirs(hparams.result_dir)

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    # Random idx for Sampling
    random_idx = random.sample(range(0, int(len(mnist_train))+1), hparams.epochs)

    Total_losses = []
    for epoch in range(hparams.epochs):
        mean_likelihood = 0.0
        num_minibatches = 0

        for batch_idx, (emb,_) in tqdm(enumerate(embs_loader)):
            emb = emb.to(device)

            u, log_det = model.forward(emb, y=None)
            prior_logprob = model.logprob(u)
            log_prob = -torch.mean(prior_logprob.sum(1) + log_det)

            opt.zero_grad()
            log_prob.backward()
            opt.step()

            mean_likelihood -= log_prob
            num_minibatches += 1

            if batch_idx == random_idx[epoch]:
                x_batch_val_tmp = autoencoder.decoder(emb)
                x_batch_val = torch.sigmoid(x_batch_val_tmp).cpu()

                x_hat_emb, d = model.backward(u, return_step=False)
                x_hat = autoencoder.decoder(x_hat_emb)
                # visualize reconstructed result at the beginning of each epoch
                x_concat = torch.cat([x_batch_val.view(-1, 1, 28, 28), x_hat.view(-1, 1, 28, 28)], dim=3)
                save_image(x_concat, './%s/reconstructed-%d.png' % (hparams.result_dir, epoch + 1))

        mean_likelihood /= num_minibatches
        print('Epoch {} completed. Log Likelihood: {}'.format(epoch, mean_likelihood))
        Total_losses.append(mean_likelihood)

        if (epoch+1) % 5 == 0:

            save_path = os.path.join(hparams.save_dir, '{}.pt'.format(epoch+1))
            torch.save(model.state_dict(), save_path)

            with torch.no_grad(): 
                z_random_sample = np.random.normal(0,hparams.temperature,[hparams.batch_size, hparams.n_input])
                z_random_sample = torch.tensor(z_random_sample).float()

                random_res_emb,_ = model.backward(z_random_sample, return_step=False) 
                random_res = autoencoder.decoder(random_res_emb)
                save_image(random_res.view(-1, 1, 28, 28), './%s/random_sampled-%d.png' % (hparams.result_dir, epoch + 1))

    total_loss = [loss.detach().cpu().numpy() for loss in Total_losses]
    x_plt = range(len(total_loss))

    plt.plot(x_plt, total_loss, label='Total Loss')

    plt.title('RealNVP(AE) Loss')
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./%s/loss.png' % (hparams.result_dir))




if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Datasets
    PARSER.add_argument('--n_input', type=int, default='20', help='The dim of input')
    PARSER.add_argument('--batch_size', type=int, default='100', help='Batch size for each epoch')

    # Adam
    PARSER.add_argument('--learning_rate', type=float, default='1e-4', help='learning_rate of Adam')
    PARSER.add_argument('--weight_decay', type=float, default='1e-5', help='Weight_decay of Adam')

    # AE Parameters
    PARSER.add_argument('--embedding_dim', type=int, default='20', help='The output dim of AE')
    PARSER.add_argument('--AE_pretrained_dir', type=str, default='./mnist_ae_realnvp/checkPoint/AE_50.pt', help='AE pretrained model dir')

    # RealNVP Parameters
    PARSER.add_argument('--realnvp_h_dim', type=int, default='200', help='Size of the hidden layers in each coupling layer in RealNVP')
    PARSER.add_argument('--flow_n', type=int, default='9', help='Number of affine coupling layers in RealNVP')

    # Training
    PARSER.add_argument('--use_cuda', type=bool, default=False, help='Device')
    PARSER.add_argument('--epochs', type=int, default='50', help='The Training epochs')

    PARSER.add_argument('--save_dir', type=str, default='./mnist_ae_realnvp/checkPoint', help='Path to store model parameters')
    PARSER.add_argument('--result_dir', type=str, default='./mnist_ae_realnvp/AERealNVP_Result', help='Path to store model results')

    # Sampling
    PARSER.add_argument('--temperature', type=float, default='1', help='Annealing parameters for Gaussian sampling')

    HPARAMS = PARSER.parse_args()
    main(HPARAMS)

