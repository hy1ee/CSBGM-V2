import numpy as np
import copy

import torch 
import torch.optim as optim
from mnist_vae.src.model_def import VAE
from mnist_nice.src.nice import NICE
from mnist_realnvp.src import real_nvp
from mnist_ae_realnvp.src.train_ae import AutoEncoder
from mnist_vae_nvp.src.model_def import VAE_NVP

import ast
import logging

import utils
import torch.nn.functional as F
# Specific Models
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lasso
from cvxopt import matrix


logging.basicConfig(filename='./variable_log.log', level=logging.DEBUG)


def lasso_estimator(A_val, y_batch_val, hparams):
    x_hat_batch = []
    for i in range(hparams.batch_size):
        y_val = y_batch_val[i]
        

        lasso_est = Lasso(alpha=hparams.lmbd)
        lasso_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
        x_hat = lasso_est.coef_

        # print(x_hat.shape)

        x_hat = np.reshape(x_hat, [-1])

        
        x_hat = np.maximum(np.minimum(x_hat, 1), 0)
        x_hat_batch.append(x_hat)
    x_hat_batch = np.asarray(x_hat_batch)
    # print(x_hat_batch.shape)
    return x_hat_batch


def vae_estimator(A, y_batch, hparams):
    loss_plt, loss_bayesian_plt = [], []
    # z = torch.randn(hparams.batch_size, hparams.n_z, requires_grad=True)
    z = utils.get_z(hparams, hparams.n_z)
    model = VAE(hparams)

    # Load pre-trained model
    checkpoint = torch.load(hparams.vae_pretrained_model_dir)
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam([z], lr = 0.01)
    
    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):
        optimizer.zero_grad()
        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()
            x_hat_batch = model.decoder(z)

            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            total_loss = torch.mean(total_loss_batch)

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_plt.append(total_loss.item())

        x_hat_batch_val = model.decoder(z)

        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch

        best_keeper.report(x_hat_batch_val, total_loss_batch_val)
    logging.info('vae_Origin_Loss: %s', loss_plt)
    return best_keeper.get_best()
            

def vae_bayesian_last_layer_estimator(A, y_batch, hparams):
    loss_plt, loss_bayesian_plt = [], []
    # z = torch.randn(hparams.batch_size, hparams.n_z, requires_grad=True)
    z = utils.get_z(hparams, hparams.n_z)
    model = VAE(hparams)

    # Load pre-trained model
    checkpoint = torch.load(hparams.vae_pretrained_model_dir)
    model.load_state_dict(checkpoint['state_dict'])


    def create_val_dict(idx_start = 5, idx_end = 8):
        val_name_s = {}
        for i in range(5,8):
            for name in ['.weight','.bias']:
                val_name_s['fc' + str(i) + name] = 0
        return val_name_s

    val_store_dict = create_val_dict()

    for name, param in model.named_parameters():
        if name in val_store_dict.keys():
            val_store_dict[name] = param # param.shape
            

    w7, b7 = val_store_dict['fc7.weight'], val_store_dict['fc7.bias']
    w7_cons, b7_cons = w7, b7

    optimizer = optim.Adam([z], lr = 0.01)
    optimizer_theta = optim.Adam([w7,b7], lr = 0.01)

    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):

        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()
            x_hat_batch = model.decoder(z)

            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            
            theta_loss_batch = torch.mean((w7 - w7_cons)**2) + torch.mean((b7 - b7_cons)**2)                                
            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_plt.append(total_loss.item())

        for j in range(hparams.max_update_iter):
            optimizer_theta.zero_grad()
            x_hat_batch = model.decoder(z)

            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            
            theta_loss_batch = torch.mean((w7 - w7_cons)**2) + torch.mean((b7 - b7_cons)**2)
                                
            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            total_loss.backward()
            optimizer_theta.step()    
            print(f"[Bayesian] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_bayesian_plt.append(total_loss.item())

        model.fc7.weight = torch.nn.Parameter(w7)
        model.fc7.bias = torch.nn.Parameter(b7)
        x_hat_batch_val = model.decoder(z)

        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        theta_loss_batch = torch.mean((w7 - w7_cons)**2) + torch.mean((b7 - b7_cons)**2)
        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch \
                            + hparams.theta_loss_weight * theta_loss_batch
        best_keeper.report(x_hat_batch_val, total_loss_batch_val)

    logging.info('vae_bayesian_last_layer_Origin_Loss: %s', loss_plt)
    logging.info('vae_bayesian_last_layer_Bayesian_Loss: %s', loss_bayesian_plt)
    return best_keeper.get_best()


def vae_bayesian_estimator(A, y_batch, hparams):
    loss_plt, loss_bayesian_plt = [], []
    # z = torch.randn(hparams.batch_size, hparams.n_z, requires_grad=True)
    z = utils.get_z(hparams, hparams.n_z)
    model = VAE(hparams)

    # Load pre-trained model
    checkpoint = torch.load(hparams.vae_pretrained_model_dir)
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam([z], lr = 0.01)
    optimizer_theta = optim.Adam(model.parameters(), lr = 0.01)
    par_origin = [param.detach().clone() for param in model.parameters()]
    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):

        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()
            x_hat_batch = model.decoder(z)

            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
             
            theta_loss_batch = 0                             
            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_plt.append(total_loss.item())

        for j in range(hparams.max_update_iter):
            optimizer_theta.zero_grad()
            x_hat_batch = model.decoder(z)

            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            
            params_temp = [param.detach().clone() for param in model.parameters()]
            squared_sum = 0
            for param_before, param_after in zip(par_origin, params_temp):
                diff = param_before - param_after
                squared_sum += torch.sum(diff ** 2)

            theta_loss_batch = torch.sqrt(squared_sum).item() 
                                
            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            total_loss.backward()
            optimizer_theta.step()    
            print(f"[Bayesian] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_bayesian_plt.append(total_loss.item())
        
        x_hat_batch_val = model.decoder(z)

        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        params_temp = [param.detach().clone() for param in model.parameters()]
        squared_sum = 0
        for param_before, param_after in zip(par_origin, params_temp):
            diff = param_before - param_after
            squared_sum += torch.sum(diff ** 2)

        theta_loss_batch = torch.sqrt(squared_sum).item() 
        # print(theta_loss_batch)

        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch \
                            + hparams.theta_loss_weight * theta_loss_batch
        best_keeper.report(x_hat_batch_val, total_loss_batch_val)

    logging.info('vae_bayesian_Origin_Loss: %s', loss_plt)
    logging.info('vae_bayesian_Bayesian_Loss: %s', loss_bayesian_plt)
    return best_keeper.get_best()


def nice_estimator(A, y_batch, hparams):
    loss_plt, loss_bayesian_plt = [], []
    # z = np.random.normal(0,hparams.temperature,[hparams.batch_size, hparams.n_input])
    # z = torch.tensor(z,requires_grad=True,dtype=torch.float)
    z = utils.get_z(hparams, hparams.n_input)
    model = NICE(hparams, data_dim=hparams.n_input)

    # Load pre-trained model
    model.load_state_dict(torch.load(hparams.nice_pretrained_model_dir))

    optimizer = optim.Adam([z], lr = 0.01)
    
    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):

        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()
            x_hat_batch = model(z, invert=True)

            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = F.mse_loss(y_batch, y_hat_batch)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            total_loss = torch.mean(total_loss_batch)

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_plt.append(total_loss.item())

        x_hat_batch_val = model(z, invert=True)

        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch

        best_keeper.report(x_hat_batch_val, total_loss_batch_val)
    logging.info('nice_Origin_Loss: %s', loss_plt)
    return best_keeper.get_best()


def nice_bayesian_estimator(A, y_batch, hparams):
    loss_plt, loss_bayesian_plt = [], []
    # z = np.random.normal(0,hparams.temperature,[hparams.batch_size, hparams.n_input])
    # z = torch.tensor(z,requires_grad=True,dtype=torch.float)
    z = utils.get_z(hparams, hparams.n_input)

    model = NICE(hparams, data_dim=hparams.n_input)
    hparams.theta_loss_weight = 5.0
    # Load pre-trained model
    model.load_state_dict(torch.load(hparams.nice_pretrained_model_dir))

    print(model)

    scale_theta = list(model.scaling_layer.parameters())[0]
    scale_theta_cons = scale_theta

    w1, b1 = list(model.coupling_layers[3].m[0].parameters())
    w2, b2 = list(model.coupling_layers[3].m[2].parameters())
    w3, b3 = list(model.coupling_layers[3].m[4].parameters())

    w4, b4 = list(model.coupling_layers[3].m[6].parameters())
    w5, b5 = list(model.coupling_layers[3].m[8].parameters())
    w6, b6 = list(model.coupling_layers[3].m[10].parameters())

    w1_cons,b1_cons,w2_cons,b2_cons,w3_cons,b3_cons,w4_cons,b4_cons,w5_cons,b5_cons,w6_cons, b6_cons = w1,b1,w2,b2,w3,b3,w4,b4,w5,b5,w6, b6


    w7, b7 = list(model.coupling_layers[2].m[0].parameters())
    w8, b8 = list(model.coupling_layers[2].m[2].parameters())
    w9, b9 = list(model.coupling_layers[2].m[4].parameters())

    w10, b10 = list(model.coupling_layers[2].m[6].parameters())
    w11, b11 = list(model.coupling_layers[2].m[8].parameters())
    w12, b12 = list(model.coupling_layers[2].m[10].parameters())

    w7_cons, b7_cons, w8_cons, b8_cons, w9_cons, b9_cons, w10_cons, b10_cons, w11_cons, b11_cons, w12_cons, b12_cons = w7, b7, w8, b8, w9, b9, w10, b10, w11, b11, w12, b12


    w13, b13 = list(model.coupling_layers[1].m[0].parameters())
    w14, b14 = list(model.coupling_layers[1].m[2].parameters())
    w15, b15 = list(model.coupling_layers[1].m[4].parameters())

    w16, b16 = list(model.coupling_layers[1].m[6].parameters())
    w17, b17 = list(model.coupling_layers[1].m[8].parameters())
    w18, b18 = list(model.coupling_layers[1].m[10].parameters())
    w13_cons, b13_cons, w14_cons, b14_cons, w15_cons, b15_cons, w16_cons, b16_cons, w17_cons, b17_cons, w18_cons, b18_cons  = w13, b13, w14, b14, w15, b15, w16, b16, w17, b17, w18, b18


    w19, b19 = list(model.coupling_layers[0].m[0].parameters())
    w20, b20 = list(model.coupling_layers[0].m[2].parameters())
    w21, b21 = list(model.coupling_layers[0].m[4].parameters())

    w22, b22 = list(model.coupling_layers[0].m[6].parameters())
    w23, b23 = list(model.coupling_layers[0].m[8].parameters())
    w24, b24 = list(model.coupling_layers[0].m[10].parameters())

    w19_cons, b19_cons, w20_cons, b20_cons, w21_cons, b21_cons, w22_cons, b22_cons, w23_cons, b23_cons, w24_cons, b24_cons = w19, b19, w20, b20, w21, b21, w22, b22, w23, b23, w24, b24

    optimizer = optim.Adam([z], lr = 0.001)

    optimizer_theta = optim.Adam([w1,b1,w2,b2,w3,b3,w4,b4,w5,b5,w6, b6,\
                                   w7, b7, w8, b8, w9, b9, w10, b10, w11, b11, w12, b12, \
                                   w13, b13, w14, b14, w15, b15, w16, b16, w17, b17, w18, b18,\
                                    w19, b19, w20, b20, w21, b21, w22, b22, w23, b23, w24, b24, \
                                    scale_theta ], lr = 0.001)

    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):
        # optimizer.zero_grad()

        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()
            x_hat_batch = model(z,invert = True)

            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            
            theta_loss_batch = torch.mean((w1 - w1_cons)**2) + torch.mean((b1 - b1_cons)**2)  \
                                + torch.mean((w2 - w2_cons)**2) + torch.mean((b2 - b2_cons)**2)  \
                                + torch.mean((w3 - w3_cons)**2) + torch.mean((b3 - b3_cons)**2)  \
                                + torch.mean((w4 - w4_cons)**2) + torch.mean((b4 - b4_cons)**2)  \
                                + torch.mean((w5 - w5_cons)**2) + torch.mean((b5 - b5_cons)**2)  \
                                + torch.mean((w6 - w6_cons)**2) + torch.mean((b6 - b6_cons)**2)  \
                                + torch.mean((w7 - w7_cons)**2) + torch.mean((b7 - b7_cons)**2)  \
                                + torch.mean((w8 - w8_cons)**2) + torch.mean((b8 - b8_cons)**2)  \
                                + torch.mean((w9 - w9_cons)**2) + torch.mean((b9 - b9_cons)**2)  \
                                + torch.mean((w10 - w10_cons)**2) + torch.mean((b10 - b10_cons)**2)  \
                                + torch.mean((w11 - w11_cons)**2) + torch.mean((b11 - b11_cons)**2)  \
                                + torch.mean((w12 - w12_cons)**2) + torch.mean((b12 - b12_cons)**2)  \
                                + torch.mean((w13 - w13_cons)**2) + torch.mean((b13 - b13_cons)**2)  \
                                + torch.mean((w14 - w14_cons)**2) + torch.mean((b14 - b14_cons)**2)  \
                                + torch.mean((w15 - w15_cons)**2) + torch.mean((b15 - b15_cons)**2)  \
                                + torch.mean((w16 - w16_cons)**2) + torch.mean((b16 - b16_cons)**2)  \
                                + torch.mean((w17 - w17_cons)**2) + torch.mean((b17 - b17_cons)**2)  \
                                + torch.mean((w18 - w18_cons)**2) + torch.mean((b18 - b18_cons)**2)  \
                                + torch.mean((w19 - w19_cons)**2) + torch.mean((b19 - b19_cons)**2)  \
                                + torch.mean((w20 - w20_cons)**2) + torch.mean((b20 - b20_cons)**2)  \
                                + torch.mean((w21 - w21_cons)**2) + torch.mean((b21 - b21_cons)**2)  \
                                + torch.mean((w22 - w22_cons)**2) + torch.mean((b22 - b22_cons)**2)  \
                                + torch.mean((w23 - w23_cons)**2) + torch.mean((b23 - b23_cons)**2)  \
                                + torch.mean((w24 - w24_cons)**2) + torch.mean((b24 - b24_cons)**2) \
                                + torch.mean((scale_theta- scale_theta_cons)**2)

            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_plt.append(total_loss.item())

        for j in range(hparams.max_update_iter):
            optimizer_theta.zero_grad()
            x_hat_batch = model(z,invert = True)
            # _, loss = model(x_hat_batch)

            # print(x_hat_batch)
            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            
            theta_loss_batch = torch.mean((w1 - w1_cons)**2) + torch.mean((b1 - b1_cons)**2)  \
                                + torch.mean((w2 - w2_cons)**2) + torch.mean((b2 - b2_cons)**2)  \
                                + torch.mean((w3 - w3_cons)**2) + torch.mean((b3 - b3_cons)**2)  \
                                + torch.mean((w4 - w4_cons)**2) + torch.mean((b4 - b4_cons)**2)  \
                                + torch.mean((w5 - w5_cons)**2) + torch.mean((b5 - b5_cons)**2)  \
                                + torch.mean((w6 - w6_cons)**2) + torch.mean((b6 - b6_cons)**2)  \
                                + torch.mean((w7 - w7_cons)**2) + torch.mean((b7 - b7_cons)**2)  \
                                + torch.mean((w8 - w8_cons)**2) + torch.mean((b8 - b8_cons)**2)  \
                                + torch.mean((w9 - w9_cons)**2) + torch.mean((b9 - b9_cons)**2)  \
                                + torch.mean((w10 - w10_cons)**2) + torch.mean((b10 - b10_cons)**2)  \
                                + torch.mean((w11 - w11_cons)**2) + torch.mean((b11 - b11_cons)**2)  \
                                + torch.mean((w12 - w12_cons)**2) + torch.mean((b12 - b12_cons)**2)  \
                                + torch.mean((w13 - w13_cons)**2) + torch.mean((b13 - b13_cons)**2)  \
                                + torch.mean((w14 - w14_cons)**2) + torch.mean((b14 - b14_cons)**2)  \
                                + torch.mean((w15 - w15_cons)**2) + torch.mean((b15 - b15_cons)**2)  \
                                + torch.mean((w16 - w16_cons)**2) + torch.mean((b16 - b16_cons)**2)  \
                                + torch.mean((w17 - w17_cons)**2) + torch.mean((b17 - b17_cons)**2)  \
                                + torch.mean((w18 - w18_cons)**2) + torch.mean((b18 - b18_cons)**2)  \
                                + torch.mean((w19 - w19_cons)**2) + torch.mean((b19 - b19_cons)**2)  \
                                + torch.mean((w20 - w20_cons)**2) + torch.mean((b20 - b20_cons)**2)  \
                                + torch.mean((w21 - w21_cons)**2) + torch.mean((b21 - b21_cons)**2)  \
                                + torch.mean((w22 - w22_cons)**2) + torch.mean((b22 - b22_cons)**2)  \
                                + torch.mean((w23 - w23_cons)**2) + torch.mean((b23 - b23_cons)**2)  \
                                + torch.mean((w24 - w24_cons)**2) + torch.mean((b24 - b24_cons)**2)  \
                                + torch.mean((scale_theta- scale_theta_cons)**2)
            
            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch
            total_loss.backward()
            optimizer_theta.step()
            print(f"[Bayesian] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_bayesian_plt.append(total_loss.item())

        x_hat_batch_val = model(z,invert = True)

        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            
        theta_loss_batch = torch.mean((w1 - w1_cons)**2) + torch.mean((b1 - b1_cons)**2)  \
                            + torch.mean((w2 - w2_cons)**2) + torch.mean((b2 - b2_cons)**2)  \
                            + torch.mean((w3 - w3_cons)**2) + torch.mean((b3 - b3_cons)**2)  \
                            + torch.mean((w4 - w4_cons)**2) + torch.mean((b4 - b4_cons)**2)  \
                            + torch.mean((w5 - w5_cons)**2) + torch.mean((b5 - b5_cons)**2)  \
                            + torch.mean((w6 - w6_cons)**2) + torch.mean((b6 - b6_cons)**2)  \
                            + torch.mean((w7 - w7_cons)**2) + torch.mean((b7 - b7_cons)**2)  \
                            + torch.mean((w8 - w8_cons)**2) + torch.mean((b8 - b8_cons)**2)  \
                            + torch.mean((w9 - w9_cons)**2) + torch.mean((b9 - b9_cons)**2)  \
                            + torch.mean((w10 - w10_cons)**2) + torch.mean((b10 - b10_cons)**2)  \
                            + torch.mean((w11 - w11_cons)**2) + torch.mean((b11 - b11_cons)**2)  \
                            + torch.mean((w12 - w12_cons)**2) + torch.mean((b12 - b12_cons)**2)  \
                            + torch.mean((w13 - w13_cons)**2) + torch.mean((b13 - b13_cons)**2)  \
                            + torch.mean((w14 - w14_cons)**2) + torch.mean((b14 - b14_cons)**2)  \
                            + torch.mean((w15 - w15_cons)**2) + torch.mean((b15 - b15_cons)**2)  \
                            + torch.mean((w16 - w16_cons)**2) + torch.mean((b16 - b16_cons)**2)  \
                            + torch.mean((w17 - w17_cons)**2) + torch.mean((b17 - b17_cons)**2)  \
                            + torch.mean((w18 - w18_cons)**2) + torch.mean((b18 - b18_cons)**2)  \
                            + torch.mean((w19 - w19_cons)**2) + torch.mean((b19 - b19_cons)**2)  \
                            + torch.mean((w20 - w20_cons)**2) + torch.mean((b20 - b20_cons)**2)  \
                            + torch.mean((w21 - w21_cons)**2) + torch.mean((b21 - b21_cons)**2)  \
                            + torch.mean((w22 - w22_cons)**2) + torch.mean((b22 - b22_cons)**2)  \
                            + torch.mean((w23 - w23_cons)**2) + torch.mean((b23 - b23_cons)**2)  \
                            + torch.mean((w24 - w24_cons)**2) + torch.mean((b24 - b24_cons)**2) \
                            + torch.mean((scale_theta- scale_theta_cons)**2)


        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch \
                            + hparams.theta_loss_weight * theta_loss_batch
        best_keeper.report(x_hat_batch_val, total_loss_batch_val)
    logging.info('nice_bayesian_Origin_Loss: %s', loss_plt)
    logging.info('nice_bayesian_Bayesian_Loss: %s', loss_bayesian_plt)
    return best_keeper.get_best()


def nice_bayesian_last_layer_estimator(A, y_batch, hparams):
    loss_plt, loss_bayesian_plt = [], []
    # z = np.random.normal(0,hparams.temperature,[hparams.batch_size, hparams.n_input])
    # z = torch.tensor(z,requires_grad=True,dtype=torch.float)

    z = utils.get_z(hparams, hparams.n_input)
    model = NICE(hparams, data_dim=hparams.n_input)

    # Load pre-trained model
    model.load_state_dict(torch.load(hparams.nice_pretrained_model_dir))


    last_linear_layer_params = list(model.coupling_layers[3].m[10].parameters())
    weight, bias = last_linear_layer_params

    scale_theta = list(model.scaling_layer.parameters())[0]

    optimizer = optim.Adam([z], lr = 0.001)
    optimizer_theta = optim.Adam([scale_theta], lr = 0.001)

    weight_cons, bias_cons = weight, bias
    scale_theta_cons = scale_theta

    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):
        # optimizer.zero_grad()

        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()
            x_hat_batch = model(z,invert = True)

            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch

            theta_loss_batch = torch.mean((weight - weight_cons)**2) + (torch.mean((bias - bias_cons)**2) + torch.mean((scale_theta- scale_theta_cons)**2) )
            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_plt.append(total_loss.item())

        for j in range(hparams.max_update_iter):
            optimizer_theta.zero_grad()
            x_hat_batch = model(z,invert = True)

            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            
            theta_loss_batch = torch.mean((weight - weight_cons)**2) + (torch.mean((bias - bias_cons)**2) + torch.mean((scale_theta- scale_theta_cons)**2) )

            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            total_loss.backward()
            optimizer_theta.step()
            print(f"[Bayesian] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_bayesian_plt.append(total_loss.item())

        model.coupling_layers[3].m[10].weight = torch.nn.Parameter(weight)
        model.coupling_layers[3].m[10].bias = torch.nn.Parameter(bias)
        with torch.no_grad():
            model.scaling_layer.log_scale_vector.data = torch.tensor(scale_theta)

        x_hat_batch_val = model(z,invert = True)

        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        theta_loss_batch = torch.mean((weight - weight_cons)**2) + (torch.mean((bias - bias_cons)**2) + torch.mean((scale_theta- scale_theta_cons)**2) )
        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch \
                            + hparams.theta_loss_weight * theta_loss_batch
        best_keeper.report(x_hat_batch_val, total_loss_batch_val)
    logging.info('nice_bayesian_last_layer_Origin_Loss: %s', loss_plt)
    logging.info('nice_bayesian_last_layer_Bayesian_Loss: %s', loss_bayesian_plt)
    return best_keeper.get_best()


def realnvp_estimator(A, y_batch, hparams):
    # hparams.zprior_weight = 0
    device = 'cpu'
    loss_plt, loss_bayesian_plt = [], []
    model = real_nvp.LinearRNVP(input_dim = hparams.n_input
                            , coupling_topology = [hparams.realnvp_h_dim]
                            , flow_n = hparams.flow_n
                            , batch_norm=True
                            , mask_type='odds'
                            , conditioning_size=None
                            , use_permutation=True
                            , single_function=True)
    model = model.to(device)

    model.load_state_dict(torch.load((hparams.realnvp_pretrained_model_dir)))
    model.eval()


    # z = np.random.normal(0,hparams.temperature,[hparams.batch_size, hparams.n_input])
    # z = torch.tensor(z).to(device).float()
    # z.requires_grad_(True)

    z = utils.get_z(hparams, hparams.n_input)
    optimizer = optim.Adam([z], lr = 0.01)
    
    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):

        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()

            x_hat_batch,_ = model.backward(z, return_step=False)
            x_hat_batch = x_hat_batch.view(hparams.batch_size, 784)
            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = F.mse_loss(y_batch, y_hat_batch)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            total_loss = torch.mean(total_loss_batch)

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_plt.append(total_loss.item())

        x_hat_batch_val,_ = model.backward(z, return_step=False)
        x_hat_batch_val = x_hat_batch_val.view(hparams.batch_size, 784)
        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch

        best_keeper.report(x_hat_batch_val, total_loss_batch_val)
    logging.info('realnvp_Origin_Loss: %s', loss_plt)
    return best_keeper.get_best()


def realnvp_bayesian_estimator(A, y_batch, hparams):
    # hparams.zprior_weight = 0
    device = 'cpu'
    model = real_nvp.LinearRNVP(input_dim = hparams.n_input
                            , coupling_topology = [hparams.realnvp_h_dim]
                            , flow_n = hparams.flow_n
                            , batch_norm=True
                            , mask_type='odds'
                            , conditioning_size=None
                            , use_permutation=True
                            , single_function=True)
    model = model.to(device)

    model.load_state_dict(torch.load((hparams.realnvp_pretrained_model_dir)))
    model.eval()
    print(model)

    for name, param in model.named_parameters():
        if name == 'flows.26.log_gamma' :# 'flows.24.s.4.weight':
            w = param
        if name == 'flows.26.beta': #'flows.24.s.4.bias':
            b = param

    w_cons, b_cons = w, b

    # z = np.random.normal(0,hparams.temperature,[hparams.batch_size, hparams.n_input])
    # z = torch.tensor(z).to(device).float()
    # z.requires_grad_(True)
    z = utils.get_z(hparams, hparams.n_input)
    optimizer = optim.Adam([z], lr = 0.01)
    
    optimizer_theta = optim.Adam([w, b], lr = 0.01)
    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):

        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()

            x_hat_batch,_ = model.backward(z, return_step=False)
            x_hat_batch = x_hat_batch.view(hparams.batch_size, 784)
            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = F.mse_loss(y_batch, y_hat_batch)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            theta_loss_batch = torch.mean((w - w_cons)**2) + torch.mean((b - b_cons)**2)                                
            
            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
        
        # model.train()
        for j in range(hparams.max_update_iter):
            optimizer_theta.zero_grad()

            x_hat_batch,_ = model.backward(z, return_step=False)
            x_hat_batch = x_hat_batch.view(hparams.batch_size, 784)
            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            
            theta_loss_batch = torch.mean((w - w_cons)**2) + torch.mean((b - b_cons)**2)

            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            print(f"[Bayesian] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            total_loss.backward()
            optimizer_theta.step()

        model.train()
        for name, param in model.named_parameters():
            if name == 'flows.26.log_gamma' :# 'flows.24.s.4.weight':
                param.data = w.clone()
            if name == 'flows.26.beta': #'flows.24.s.4.bias':
                param.data = b.clone()
        model.eval()
        x_hat_batch_val,_ = model.backward(z, return_step=False)
        x_hat_batch_val = x_hat_batch_val.view(hparams.batch_size, 784)
        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        theta_loss_batch = torch.mean((w - w_cons)**2) + torch.mean((b - b_cons)**2)
        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch \
                            + hparams.theta_loss_weight * theta_loss_batch
        print(total_loss_batch_val)
        best_keeper.report(x_hat_batch_val, total_loss_batch_val)

    return best_keeper.get_best()


def ae_realnvp_estimator(A, y_batch, hparams):
    # hparams.zprior_weight = 0
    loss_plt, loss_bayesian_plt = [], []
    device = 'cpu'
    autoencoder = AutoEncoder(hparams)
    autoencoder = autoencoder.to(device)
    autoencoder.load_state_dict(torch.load((hparams.ae_pretrained_model_dir)))

    model = real_nvp.LinearRNVP(input_dim = hparams.embedding_dim
                        , coupling_topology = [hparams.ae_realnvp_h_dim]
                        , flow_n = hparams.flow_n
                        , batch_norm=True
                        , mask_type='odds'
                        , conditioning_size=None
                        , use_permutation=True
                        , single_function=True)

    model = model.to(device)

    model.load_state_dict(torch.load((hparams.ae_realnvp_pretrained_model_dir)))
    model.eval()

    # z = np.random.normal(0,hparams.temperature,[hparams.batch_size, hparams.embedding_dim])
    # z = torch.tensor(z,requires_grad=True,dtype=torch.float)
    # z.requires_grad_(True)
    z = utils.get_z(hparams, hparams.embedding_dim)
    optimizer = optim.Adam([z], lr = 0.01)
    
    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):

        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()

            emb,d = model.backward(z)
            z_temp = autoencoder.decoder(emb)
            d_sorted = d.sort(0)[1].flip(0)
            z_temp = z_temp[d_sorted]
            x_hat_batch = torch.sigmoid(z_temp).cpu()

            # x_hat_batch = model(z, invert=True)
            x_hat_batch = x_hat_batch.view(hparams.batch_size, 784)
            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = F.mse_loss(y_batch, y_hat_batch)# torch.mean((y_batch - y_hat_batch)**2, 1) + 
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            total_loss = torch.mean(total_loss_batch)

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_plt.append(total_loss.item())

        emb,d = model.backward(torch.tensor(z).to(device).float())
        z_temp = autoencoder.decoder(emb)
        d_sorted = d.sort(0)[1].flip(0)
        z_temp = z_temp[d_sorted]
        x_hat_batch_val = torch.sigmoid(z_temp).cpu()

        x_hat_batch_val = x_hat_batch_val.view(hparams.batch_size, hparams.n_input)
        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch

        best_keeper.report(x_hat_batch_val, total_loss_batch_val)
    logging.info('ae_realnvp_Origin_Loss: %s', loss_plt)
    return best_keeper.get_best()


def ae_realnvp_bayesian_estimator(A, y_batch, hparams):

    loss_plt, loss_bayesian_plt = [], []
    device = 'cpu'
    autoencoder = AutoEncoder(hparams)
    autoencoder = autoencoder.to(device)
    autoencoder.load_state_dict(torch.load((hparams.ae_pretrained_model_dir)))
    print(autoencoder)

    model = real_nvp.LinearRNVP(input_dim = hparams.embedding_dim
                        , coupling_topology = [hparams.ae_realnvp_h_dim]
                        , flow_n = hparams.flow_n
                        , batch_norm=True
                        , mask_type='odds'
                        , conditioning_size=None
                        , use_permutation=True
                        , single_function=True)

    model = model.to(device)
    
    for name, param in autoencoder.named_parameters():
        # print(name)
        if name == 'linear2.weight':
            w1 = param
        if name == 'linear2.bias':
            b1 = param
        if name == 'convt1.weight':
            w2 = param
        if name == 'convt1.bias':
            b2 = param
        if name == 'convt2.weight':
            w3 = param
        if name == 'convt2.bias':
            b3 = param


    model.load_state_dict(torch.load((hparams.ae_realnvp_pretrained_model_dir)))
    model.eval()

    z = np.random.normal(0,hparams.temperature,[hparams.batch_size, hparams.embedding_dim])
    z = torch.tensor(z,requires_grad=True,dtype=torch.float)
    z.requires_grad_(True)
    optimizer = optim.Adam([z], lr = 0.01)
    
    optimizer_theta = optim.Adam([w3, b3], lr = 0.01)
    w1_cons, b1_cons,w2_cons, b2_cons, w3_cons, b3_cons = w1, b1, w2, b2, w3, b3

    best_keeper = utils.BestKeeper(hparams)


    for i in range(hparams.num_random_restarts):

        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()

            emb,d = model.backward(z)
            z_temp = autoencoder.decoder(emb)
            d_sorted = d.sort(0)[1].flip(0)
            z_temp = z_temp[d_sorted]
            x_hat_batch = torch.sigmoid(z_temp).cpu()

            x_hat_batch = x_hat_batch.view(hparams.batch_size, hparams.n_input)
            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = F.mse_loss(y_batch, y_hat_batch)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            

            theta_loss_batch = torch.mean((w1 - w1_cons)**2) + (torch.mean((b1 - b1_cons)**2)) \
                        + torch.mean((w2 - w2_cons)**2) + (torch.mean((b2 - b2_cons)**2)) \
                        + torch.mean((w3 - w3_cons)**2) + (torch.mean((b3 - b3_cons)**2))
            
            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_plt.append(total_loss.item())

        for j in range(hparams.max_update_iter):
            optimizer_theta.zero_grad()

            emb,d = model.backward(z)
            z_temp = autoencoder.decoder(emb)
            d_sorted = d.sort(0)[1].flip(0)
            z_temp = z_temp[d_sorted]
            x_hat_batch = torch.sigmoid(z_temp).cpu()

            x_hat_batch = x_hat_batch.view(hparams.batch_size, hparams.n_input)
            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            
            
            theta_loss_batch = torch.mean((w1 - w1_cons)**2) + (torch.mean((b1 - b1_cons)**2)) \
                                + torch.mean((w2 - w2_cons)**2) + (torch.mean((b2 - b2_cons)**2)) \
                                + torch.mean((w3 - w3_cons)**2) + (torch.mean((b3 - b3_cons)**2))
            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch
            
            total_loss.backward()
            optimizer_theta.step()
            print(f"[Bayesian] num_restart:{i} | iter:{j} | total_loss:{total_loss} | theta_loss:{hparams.theta_loss_weight * theta_loss_batch}")
            loss_bayesian_plt.append(total_loss.item())

        autoencoder.linear2.weight.data,autoencoder.linear2.bias.data  = torch.nn.Parameter(w1), torch.nn.Parameter(b1)
        autoencoder.convt1.weight.data,autoencoder.convt1.bias.data  = torch.nn.Parameter(w2), torch.nn.Parameter(b2)
        autoencoder.convt2.weight.data,autoencoder.convt2.bias.data  = torch.nn.Parameter(w3), torch.nn.Parameter(b3)

        emb,d = model.backward(torch.tensor(z).to(device).float())
        z_temp = autoencoder.decoder(emb)
        d_sorted = d.sort(0)[1].flip(0)
        z_temp = z_temp[d_sorted]
        x_hat_batch_val = torch.sigmoid(z_temp).cpu()

        x_hat_batch_val = x_hat_batch_val.view(hparams.batch_size, hparams.n_input)
        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        theta_loss_batch = torch.mean((w1 - w1_cons)**2) + (torch.mean((b1 - b1_cons)**2)) \
                                + torch.mean((w2 - w2_cons)**2) + (torch.mean((b2 - b2_cons)**2)) \
                                + torch.mean((w3 - w3_cons)**2) + (torch.mean((b3 - b3_cons)**2))

        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch \
                            + hparams.theta_loss_weight * theta_loss_batch
        
        best_keeper.report(x_hat_batch_val, total_loss_batch_val)

    logging.info('vae_nvp_bayesian_Origin_Loss: %s', loss_plt)
    logging.info('vae_nvp_bayesian_Bayesian_Loss: %s', loss_bayesian_plt)
    return best_keeper.get_best()
    # return x_hat_batch_val.detach().numpy()


def vae_nvp_estimator(A, y_batch, hparams):
    loss_plt, loss_bayesian_plt = [], []
    # z = torch.randn(hparams.batch_size, hparams.n_z, requires_grad=True)
    z = utils.get_z(hparams, hparams.n_z)
    model = VAE_NVP(hparams)

    model.load_state_dict(torch.load(hparams.vae_nvp_pretrained_model_dir))
    model.eval()
    optimizer = optim.Adam([z], lr = 0.01)
    
    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):
        optimizer.zero_grad()
        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()

            # decoder
            random_res = model.decoder(z)

            x_hat_batch = random_res

            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            total_loss = torch.mean(total_loss_batch)

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_plt.append(total_loss.item())
       #  z_sampled_flow_in, _ = model.flow_forward(z)
        # z_transfrom = model.flow_backward(z)
        # decoder
        random_res = model.decoder(z)
        x_hat_batch_val = random_res# .view(-1, 1, 28, 28)

        # x_hat_batch_val = model.decoder(z)

        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch

        best_keeper.report(x_hat_batch_val, total_loss_batch_val)
    logging.info('vae_nvp_Origin_Loss: %s', loss_plt)
    return best_keeper.get_best()


def vae_nvp_bayesian_estimator(A, y_batch, hparams):

    loss_plt, loss_bayesian_plt = [], []
    # z = torch.randn(hparams.batch_size, hparams.n_z, requires_grad=True)
    z = utils.get_z(hparams, hparams.n_z)
    model = VAE_NVP(hparams)

    model.load_state_dict(torch.load(hparams.vae_nvp_pretrained_model_dir))
    # print(model)
    for name, param in model.named_parameters():
        if name == 'fc7.weight':
            w7 = param
        if name == 'fc7.bias':
            b7 = param

    w7_cons, b7_cons = w7, b7


    optimizer = optim.Adam([z], lr = 0.01)
    optimizer_theta = optim.Adam([w7,b7], lr = 0.01)
    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):
        # optimizer.zero_grad()
        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()
            x_hat_batch = model.decoder(z)
            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            
            theta_loss_batch = torch.mean((w7 - w7_cons)**2) + torch.mean((b7 - b7_cons)**2) 
            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            total_loss.backward()
            optimizer.step()
            print(f"[Origin] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_plt.append(total_loss.item())

        for j in range(hparams.max_update_iter):
            optimizer_theta.zero_grad()

            x_hat_batch = model.decoder(z)
            y_hat_batch = torch.matmul(x_hat_batch.double(), A.double())

            # Define Loss
            m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
            m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
            zp_loss_batch = torch.sum(z**2, 1)

            # define total loss
            total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch
            
            theta_loss_batch = torch.mean((w7 - w7_cons)**2) + torch.mean((b7 - b7_cons)**2) 
            total_loss = torch.mean(total_loss_batch) + hparams.theta_loss_weight * theta_loss_batch

            total_loss.backward()
            optimizer_theta.step()
            print(f"[Bayesian] num_restart:{i} | iter:{j} | total_loss:{total_loss}")
            loss_bayesian_plt.append(total_loss.item())


        random_res = model.decoder(z)
        x_hat_batch_val = random_res

        y_hat_batch = torch.matmul(x_hat_batch_val.double(), A.double())
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        theta_loss_batch = torch.mean((w7 - w7_cons)**2) + torch.mean((b7 - b7_cons)**2)
        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                                + hparams.mloss2_weight * m_loss2_batch \
                                + hparams.zprior_weight * zp_loss_batch \
                                + hparams.theta_loss_weight * theta_loss_batch
        
        best_keeper.report(x_hat_batch_val, total_loss_batch_val)
    logging.info('vae_nvp_bayesian_Origin_Loss: %s', loss_plt)
    logging.info('vae_nvp_bayesian_Bayesian_Loss: %s', loss_bayesian_plt)
    return best_keeper.get_best()

