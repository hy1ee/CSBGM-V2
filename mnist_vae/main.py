import os
import numpy as np
import model_def
import utils

def main(hparams):
    # 
    utils.print_hparams(hparams)








if __name__ == '__main__':
    HPARAMS = model_def.Hparams()

    HPARAMS.num_samples = 60000
    HPARAMS.learning_rate = 0.001
    HPARAMS.batch_size = 100
    HPARAMS.training_epochs = 100
    HPARAMS.summary_epoch = 1
    HPARAMS.ckpt_epoch = 5

