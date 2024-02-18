import numpy as np
import copy

import torch 
import torch.optim as optim
from mnist_vae.src.model_def import VAE
import utils

# Specific Models
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lasso
from cvxopt import matrix
# from l1regls import l1regls


def lasso_estimator(hparams):  # pylint: disable = W0613
    """LASSO estimator"""
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        for i in range(hparams.batch_size):
            y_val = y_batch_val[i]
            
            # x_hat = utils.solve_lasso(A_val, y_val, hparams)
            if hparams.lasso_solver == 'sklearn':
                lasso_est = Lasso(alpha=hparams.lmbd)
                lasso_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
                x_hat = lasso_est.coef_
                x_hat = np.reshape(x_hat, [-1])
            if hparams.lasso_solver == 'cvxopt':
                A_mat = matrix(A_val.T)
                y_mat = matrix(y_val)
                x_hat_mat = l1regls(A_mat, y_mat)
                x_hat = np.asarray(x_hat_mat)
                x_hat = np.reshape(x_hat, [-1])

            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator


def vae_estimator(A, y_batch, hparams):
    z = torch.randn(hparams.batch_size, hparams.n_z, requires_grad=True)
    model = VAE(hparams)

    # Load pre-trained model
    dir = './mnist_vae/checkPoint/model_best.pth' # best model
    checkpoint = torch.load(hparams.mnist_dir)
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam([z], lr = 0.01)
    
    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):
        # optimizer.zero_grad()

        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()
            x_hat_batch = model.decoder(z)

            y_hat_batch = torch.matmul(x_hat_batch, A)

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

            # print(f"rr {i} iter {j} total_loss {total_loss} m_loss1 {m_loss1_batch} \
            #       m_loss2 {m_loss2_batch} zp_loss {zp_loss_batch}")
        # print(z)
        x_hat_batch_val = model.decoder(z)

        y_hat_batch = torch.matmul(x_hat_batch_val, A)
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch

        best_keeper.report(x_hat_batch_val, total_loss_batch_val)

    return best_keeper.get_best()
        # best_keeper.report(x_hat_batch_val, total_loss_batch_val)
            

def vae_bayesian_estimator(A, y_batch, hparams):
    z = torch.randn(hparams.batch_size, hparams.n_z, requires_grad=True)
    model = VAE(hparams)

    # Load pre-trained model
    dir = './mnist_vae/checkPoint/model_best.pth' # best model
    checkpoint = torch.load(hparams.mnist_dir)
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
            
    # print(val_store_dict.keys())
    # w,b = val_store_dict['fc7.weight'], val_store_dict['fc7.bias']
    # w6, b6 = val_store_dict['fc6.weight'], val_store_dict['fc6.bias']
    w7, b7 = val_store_dict['fc7.weight'], val_store_dict['fc7.bias']

    # w6_cons, b6_cons = w6, b6
    w7_cons, b7_cons = w7, b7
    # val_store_dict_const = [val_store_dict['fc7.weight'], val_store_dict['fc7.bias']]
    
    # var_list_theta = [val_store_dict['fc7.weight'], val_store_dict['fc7.bias']]

    optimizer = optim.Adam([z], lr = 0.01)
    optimizer_theta = optim.Adam([w7,b7], lr = 0.01)


    best_keeper = utils.BestKeeper(hparams)

    for i in range(hparams.num_random_restarts):
        # optimizer.zero_grad()

        for j in range(hparams.max_update_iter):
            optimizer.zero_grad()
            x_hat_batch = model.decoder(z)

            y_hat_batch = torch.matmul(x_hat_batch, A)

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

            # print(f"rr {i} iter {j} total_loss {total_loss} m_loss1 {m_loss1_batch} \
            #       m_loss2 {m_loss2_batch} zp_loss {zp_loss_batch}")
        # print(z)
        # optimizer_theta.zero_grad()
        for j in range(hparams.max_update_iter):
            optimizer_theta.zero_grad()
            x_hat_batch = model.decoder(z)

            y_hat_batch = torch.matmul(x_hat_batch, A)

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

        # model.fc6.weight = torch.nn.Parameter(w6)
        # model.fc6.bias = torch.nn.Parameter(b6)
        model.fc7.weight = torch.nn.Parameter(w7)
        model.fc7.bias = torch.nn.Parameter(b7)
        x_hat_batch_val = model.decoder(z)

        y_hat_batch = torch.matmul(x_hat_batch_val, A)
        m_loss1_batch = torch.mean(torch.abs(y_batch - y_hat_batch), 1)
        m_loss2_batch = torch.mean((y_batch - y_hat_batch)**2, 1)
        zp_loss_batch = torch.sum(z**2, 1)

        # theta_loss_batch = torch.mean((w7 - w7_cons)**2) + torch.mean((b7 - b7_cons)**2)


        theta_loss_batch = torch.mean((w7 - w7_cons)**2) + torch.mean((b7 - b7_cons)**2)
        # define total loss
        total_loss_batch_val = hparams.mloss1_weight * m_loss1_batch \
                            + hparams.mloss2_weight * m_loss2_batch \
                            + hparams.zprior_weight * zp_loss_batch \
                            + hparams.theta_loss_weight * theta_loss_batch
        best_keeper.report(x_hat_batch_val, total_loss_batch_val)

    return best_keeper.get_best()
        # best_keeper.report(x_hat_batch_val, total_loss_batch_val)



def nice_estimator(A, y_batch, hparams):
    return