import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Uniform
import numpy as np
import copy

import torch
from torch import nn, distributions

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mnist_realnvp.src.real_nvp import LinearRNVP


class VAE_NVP(nn.Module):
    def __init__(self, hparams):
        super(VAE_NVP, self).__init__()

        self.n_input = hparams.n_input
        self.n_z = hparams.n_z
        self.h_dim = hparams.vae_h_dim

        self.flow = LinearRNVP(input_dim = hparams.n_z
                                , coupling_topology = [hparams.vae_nvp_h_dim]
                                , flow_n = hparams.flow_n
                                , batch_norm=True
                                , mask_type='odds'
                                , conditioning_size=None
                                , use_permutation=True
                                , single_function=True)

        # Encoder
        self.fc1 = nn.Linear(self.n_input, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.h_dim)
        self.fc3_mean = nn.Linear(self.h_dim, self.n_z)
        self.fc4_log_var = nn.Linear(self.h_dim, self.n_z)

        # Decoder
        self.fc5 = nn.Linear(self.n_z, self.h_dim)
        self.fc6 = nn.Linear(self.h_dim, self.h_dim)
        self.fc7 = nn.Linear(self.h_dim, self.n_input)

    def forward(self, x):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training model [b, batch_size, 1, 28, 28]
        :return: the result of our training model
        """
        batch_size = x.shape[0]  # 每一批含有的样本的个数
        # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
        # tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，
        # 返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
        x = x.view(batch_size, self.n_input)  # 一行代表一个样本

        # encoder
        z_mean, z_log_sigma_sq = self.encoder(x)

        z_sampled = z_mean + torch.randn_like(torch.exp(z_log_sigma_sq * 0.5))

        z_sampled_flow_in, likelihood = self.flow_forward(z_sampled)
        # z_transfrom = self.flow_backward(z_sampled_flow_in)

        # decoder
        x_hat = self.decoder(z_sampled_flow_in)
        # reshape
        x_hat = x_hat.view(batch_size, 1, 28, 28)
        return x_hat, z_mean, z_log_sigma_sq, likelihood



    def encoder(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        hidden1 = F.softplus(self.fc1(x))
        hidden2 = F.softplus(self.fc2(hidden1))
        z_mean = self.fc3_mean(hidden2)
        z_log_sigma_sq = self.fc4_log_var(hidden2)

        return z_mean, z_log_sigma_sq

    def decoder(self, z):
        """
        Given a sampled z, decoder/generator it back to image
        """
        hidden1 = F.softplus(self.fc5(z))
        hidden2 = F.softplus(self.fc6(hidden1))
        x_hat = torch.sigmoid(self.fc7(hidden2))  # 图片数值取值为[0,1]，不宜用ReLU
        return x_hat


    def flow_forward(self, z_sample):
        # z_flow_in, likelihood = self.flow(z_sample, y = None) # z_sample: bs, n_z
        
        u, log_det = self.flow.forward(z_sample, y=None)
        prior_logprob = self.flow.logprob(u)
        log_prob = -torch.mean(prior_logprob.sum(1) + log_det)

        return u, log_prob
    
    def flow_backward(self, z_flow_in):
        z_transform,_ = self.flow.backward(z_flow_in, return_step=False)
        return z_transform