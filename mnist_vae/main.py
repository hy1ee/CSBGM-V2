import os
import numpy as np
import model_def
import utils
import torch


def main(hparams):
    # 
    utils.print_hparams(hparams)

    # encode
    x_ph = torch.FloatTensor(None, hparams.n_input)
    encoder = model_def.Encoder(hparams)
    z_mean, z_log_sigma_sq = encoder(x_ph)

    # sample
    eps = torch.randn((hparams.batch_size, hparams.n_z), dtype=torch.float32)
    z_sigma = torch.sqrt(torch.exp(z_log_sigma_sq))
    z = z_mean + z_sigma * eps

    # reconstruct
    generator = model_def.Generator(hparams)
    logits, x_reconstr_mean = generator(z)

    # generator sampler
    z_ph = torch.FloatTensor(None, hparams.n_z)
    _, x_sample = generator(z_ph)

    # define loss
    total_loss = model_def.get_loss(x_ph, logits, z_mean, z_log_sigma_sq)





if __name__ == '__main__':
    HPARAMS = model_def.Hparams()

    HPARAMS.num_samples = 60000
    HPARAMS.learning_rate = 0.001
    HPARAMS.batch_size = 100
    HPARAMS.training_epochs = 100
    HPARAMS.summary_epoch = 1
    HPARAMS.ckpt_epoch = 5

    main(HPARAMS)
