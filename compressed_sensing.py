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

    if hparams.if_show_images:
        model_name = [  
                        'origin'
                        , 'lasso'
                        # , 'vae'
                        # , 'vae_bayesian_last_layer'
                        # , 'vae_bayesian'

                        # , 'nice'
                        # , 'nice_bayesian'
                        # , 'nice_bayesian_last_layer'

                        # , 'realnvp'
                        # , 'realnvp_bayesian'
                        , 'ae_realnvp'
                        , 'ae_realnvp_bayesian'
                        # , 'vae_nvp'
                        # , 'vae_nvp_bayesian'
                
                        ]

        x_batch, _ = next(iter(mnist_train))
        x_batch_numpy = x_batch.squeeze().numpy()
        x_batch = x_batch.view(hparams.batch_size, hparams.n_input)
        
        A = np.random.randn(hparams.n_input, hparams.num_measurements) 
        A = torch.tensor(A, dtype=torch.float32)
        noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, hparams.num_measurements)
        y_batch = np.matmul(x_batch, A) + noise_batch 

        estimators_list = [x_batch_numpy]

        for name in model_name:
            x_batch_hat = None
            if name == 'origin':    
                continue

            if name == 'lasso':
                x_batch_hat = mnist_estimators.lasso_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")
            
            if name == 'vae':
                x_batch_hat = mnist_estimators.vae_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")

            if name == 'vae_bayesian_last_layer':
                x_batch_hat = mnist_estimators.vae_bayesian_last_layer_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")

            if name == 'vae_bayesian':
                x_batch_hat = mnist_estimators.vae_bayesian_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")

            if name == 'nice':
                x_batch_hat = mnist_estimators.nice_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")

            if name == 'nice_bayesian':
                x_batch_hat = mnist_estimators.nice_bayesian_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")

            if name == 'nice_bayesian_last_layer':
                x_batch_hat = mnist_estimators.nice_bayesian_last_layer_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")


            if name == 'realnvp':
                x_batch_hat = mnist_estimators.realnvp_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")

            if name == 'realnvp_bayesian':
                x_batch_hat = mnist_estimators.realnvp_bayesian_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")

            if name == 'ae_realnvp':
                x_batch_hat = mnist_estimators.ae_realnvp_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")

            if name == 'ae_realnvp_bayesian':
                x_batch_hat = mnist_estimators.ae_realnvp_bayesian_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")

            if name == 'vae_nvp':
                x_batch_hat = mnist_estimators.vae_nvp_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")

            if name == 'vae_nvp_bayesian':
                x_batch_hat = mnist_estimators.vae_nvp_bayesian_estimator(A, y_batch, hparams)
                x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                print(f"{name} model Completed")


            estimators_list.append(x_batch_hat)

        assert len(estimators_list) == len(model_name)

        mse_values = utils.mse_between_elements(estimators_list)
        for idx, mse in enumerate(mse_values, start=2):
            print(f"{model_name[idx-1]} MSE:", mse)

        utils.show_images(estimators_list,model_name, hparams)


    lasso_l2_loss, lasso_measure_loss = 0,0
    vae_l2_loss, vae_measure_loss = 0,0

    vae_bayesian_last_layer_l2_loss, vae_bayesian_last_layer_measure_loss = 0,0
    vae_bayesian_l2_loss, vae_bayesian_measure_loss = 0,0

    nice_l2_loss, nice_measure_loss = 0,0
    nice_bayesian_l2_loss, nice_bayesian_measure_loss = 0,0
    nice_bayesian_last_layer_l2_loss, nice_bayesian_last_layer_measure_loss = 0,0

    realnvp_l2_loss, realnvp_measure_loss = 0,0
    realnvp_bayesian_l2_loss, realnvp_bayesian_measure_loss = 0,0

    ae_realnvp_l2_loss, ae_realnvp_measure_loss = 0,0
    ae_realnvp_bayesian_l2_loss, ae_realnvp_bayesian_measure_loss = 0,0

    vae_nvp_l2_loss, vae_nvp_measure_loss = 0,0
    vae_nvp_bayesian_last_layer_l2_loss, vae_nvp_bayesian_last_layer_measure_loss = 0,0
    vae_nvp_bayesian_l2_loss, vae_nvp_bayesian_measure_loss = 0,0


    step = 0
    for batch_idx, (x_batch, _) in enumerate(mnist_train):
        step += 1
        x_batch = x_batch.view(hparams.batch_size, hparams.n_input) 

        A = np.random.randn(hparams.n_input, hparams.num_measurements)
        A = torch.tensor(A, dtype=torch.float32)
        noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, hparams.num_measurements)
        y_batch = np.matmul(x_batch, A) + noise_batch

        for name in model_name:
            x_batch_hat = None
            if name == 'origin':    
                continue

            if name == 'lasso':
                x_batch_hat = mnist_estimators.lasso_estimator(A, y_batch, hparams)
                lasso_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                lasso_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")
            
            if name == 'vae':
                x_batch_hat = mnist_estimators.vae_estimator(A, y_batch, hparams)
                vae_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                vae_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

            if name == 'vae_bayesian_last_layer':
                x_batch_hat = mnist_estimators.vae_bayesian_last_layer_estimator(A, y_batch, hparams)
                vae_bayesian_last_layer_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                vae_bayesian_last_layer_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

            if name == 'vae_bayesian':
                x_batch_hat = mnist_estimators.vae_bayesian_estimator(A, y_batch, hparams)
                vae_bayesian_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                vae_bayesian_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

            if name == 'nice':
                x_batch_hat = mnist_estimators.nice_estimator(A, y_batch, hparams)
                nice_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                nice_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

            if name == 'nice_bayesian':
                x_batch_hat = mnist_estimators.nice_bayesian_estimator(A, y_batch, hparams)
                nice_bayesian_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                nice_bayesian_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

            if name == 'nice_bayesian_last_layer':
                x_batch_hat = mnist_estimators.nice_bayesian_last_layer_estimator(A, y_batch, hparams)
                nice_bayesian_last_layer_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                nice_bayesian_last_layer_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")


            if name == 'realnvp':
                x_batch_hat = mnist_estimators.realnvp_estimator(A, y_batch, hparams)
                realnvp_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                realnvp_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

            if name == 'realnvp_bayesian':
                x_batch_hat = mnist_estimators.realnvp_bayesian_estimator(A, y_batch, hparams)
                realnvp_bayesian_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                realnvp_bayesian_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

            if name == 'ae_realnvp':
                x_batch_hat = mnist_estimators.ae_realnvp_estimator(A, y_batch, hparams)
                ae_realnvp_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                ae_realnvp_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

            if name == 'ae_realnvp_bayesian':
                x_batch_hat = mnist_estimators.ae_realnvp_bayesian_estimator(A, y_batch, hparams)
                ae_realnvp_bayesian_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                ae_realnvp_bayesian_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

            if name == 'vae_nvp':
                x_batch_hat = mnist_estimators.vae_nvp_estimator(A, y_batch, hparams)
                vae_nvp_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                vae_nvp_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

            if name == 'vae_nvp_bayesian_last_layer':
                x_batch_hat = mnist_estimators.vae_nvp_bayesian_last_layer_estimator(A, y_batch, hparams)
                # x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                vae_nvp_bayesian_last_layer_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                vae_nvp_bayesian_last_layer_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

            if name == 'vae_nvp_bayesian':
                x_batch_hat = mnist_estimators.vae_nvp_bayesian_estimator(A, y_batch, hparams)
                # x_batch_hat = x_batch_hat.reshape(hparams.batch_size, 1, 28, 28)
                vae_nvp_bayesian_l2_loss += utils.get_l2_loss(x_batch_hat, x_batch)
                vae_nvp_bayesian_measure_loss += utils.get_measurement_loss(x_batch_hat, A, y_batch)
                print(f"{name} model Completed")

        if step >= hparams.steps:
            break

    
    print(f"Lasso | L2_loss:{np.mean(lasso_l2_loss)} | Measurement_loss:{np.mean(lasso_measure_loss)}")
    
    print(f"vae | L2_loss:{np.mean(vae_l2_loss)} | Measurement_loss:{np.mean(vae_measure_loss)}")
    print(f"vae_bayesian_last_layer | L2_loss:{np.mean(vae_bayesian_last_layer_l2_loss)} | Measurement_loss:{np.mean(vae_bayesian_last_layer_measure_loss)}")
    print(f"vae_bayesian | L2_loss:{np.mean(vae_bayesian_l2_loss)} | Measurement_loss:{np.mean(vae_bayesian_measure_loss)}")
    
    print(f"nice | L2_loss:{np.mean(nice_l2_loss)} | Measurement_loss:{np.mean(nice_measure_loss)}")
    print(f"nice_bayesian | L2_loss:{np.mean(nice_bayesian_l2_loss)} | Measurement_loss:{np.mean(nice_bayesian_measure_loss)}")
    print(f"nice_bayesian_last_layer | L2_loss:{np.mean(nice_bayesian_last_layer_l2_loss)} | Measurement_loss:{np.mean(nice_bayesian_last_layer_measure_loss)}")
    
    print(f"realnvp | L2_loss:{np.mean(realnvp_l2_loss)} | Measurement_loss:{np.mean(realnvp_measure_loss)}")
    print(f"realnvp_bayesian | L2_loss:{np.mean(realnvp_bayesian_l2_loss)} | Measurement_loss:{np.mean(realnvp_bayesian_measure_loss)}")
    
    print(f"ae_realnvp | L2_loss:{np.mean(ae_realnvp_l2_loss)} | Measurement_loss:{np.mean(ae_realnvp_measure_loss)}")
    print(f"ae_realnvp_bayesian | L2_loss:{np.mean(ae_realnvp_bayesian_l2_loss)} | Measurement_loss:{np.mean(ae_realnvp_bayesian_measure_loss)}")
    
    print(f"vae_nvp | L2_loss:{np.mean(vae_nvp_l2_loss)} | Measurement_loss:{np.mean(vae_nvp_measure_loss)}")
    print(f"vae_nvp_bayesian_last_layer | L2_loss:{np.mean(vae_nvp_bayesian_last_layer_l2_loss)} | Measurement_loss:{np.mean(vae_nvp_bayesian_last_layer_measure_loss)}")
    print(f"vae_nvp_bayesian | L2_loss:{np.mean(vae_nvp_bayesian_l2_loss)} | Measurement_loss:{np.mean(vae_nvp_bayesian_measure_loss)}")
    


if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Training Config
    PARSER.add_argument('--num_random_restarts', type=float, default=1, help='num of training random starts')
    PARSER.add_argument('--max_update_iter', type=float, default=1000, help='num of training epochs per session')
    PARSER.add_argument('--mloss1_weight', type=float, default=0.0, help='L1 measurement loss weight')
    PARSER.add_argument('--mloss2_weight', type=float, default=1.0, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior_weight', type=float, default=0.1, help='weight on z prior')

    # bayesian inference
    PARSER.add_argument('--theta_loss_weight', type=float, default=0.1, help='weight on z theta')

    # input
    PARSER.add_argument('--batch_size', type=int, default=1, help='How many examples are processed together')
    PARSER.add_argument('--n_input', type=int, default=784, help='The dim of input images')

    PARSER.add_argument('--num_measurements', type=int, default=50, help='num measurements(A)')
    PARSER.add_argument('--noise_std', type=float, default=0.1, help='std of noise(n)')

    # get z
    PARSER.add_argument('--get_z_method', type=str, default='Gaussian', help='Or Fixed, Modify in the utils file')

    # Lasso model Parameters
    PARSER.add_argument('--lmbd', type=float, default='0.01', help='The lmbd of Lasso')

    # VAE model Parameters
    PARSER.add_argument('--n_z', type=int, default='20', help='The dim of latent z')
    PARSER.add_argument('--vae_h_dim', type=int, default='500', help='The dim of hidden in VAE')
    PARSER.add_argument('--vae_pretrained_model_dir', type=str, default='./mnist_vae/checkPoint/model_best.pth', help='The dim of hidden')

    # Flow model Parameters
    PARSER.add_argument('--temperature', type=float, default='0.8', help='Annealing parameters for Gaussian sampling')

    # NICE model Parameters
    PARSER.add_argument('--use_cuda', type=bool, default=False, help='The shape of mnist')
    PARSER.add_argument('--nice_h_dim', type=int, default='1000', help='The hidden linear dim of Layer in NICE')
    PARSER.add_argument('--num_net_layers', type=int, default='6', help='The num of net layers in NICE')
    PARSER.add_argument('--num_coupling_layers', type=int, default='4', help='The num of coupling layers in NICE')
    PARSER.add_argument('--nice_pretrained_model_dir', type=str, default='./mnist_nice/checkPoint/20.pt', help='The dim of hidden')


    # RealNVP Parameters
    PARSER.add_argument('--realnvp_h_dim', type=int, default='1000', help='Size of the hidden layers in each coupling layer in RealNVP')
    PARSER.add_argument('--flow_n', type=int, default='9', help='Number of affine coupling layers in RealNVP')
    PARSER.add_argument('--realnvp_pretrained_model_dir', type=str, default='./mnist_realnvp/checkPoint/50.pt', help='The pretrained model dir of RealNVP')
    
    # AE_RealNVP Parameters
    PARSER.add_argument('--embedding_dim', type=int, default='20', help='The output dim of AE')
    PARSER.add_argument('--ae_realnvp_h_dim', type=int, default='200', help='Size of the hidden layers in each coupling layer in AE_RealNVP')
    PARSER.add_argument('--ae_pretrained_model_dir', type=str, default='./mnist_ae_realnvp/checkPoint/AE_50.pt', help='The pretrained model dir of AE')
    PARSER.add_argument('--ae_realnvp_pretrained_model_dir', type=str, default='./mnist_ae_realnvp/checkPoint/50.pt', help='The pretrained model dir of AE_RealNVP')
    
    # Vae_NVP Parameters
    PARSER.add_argument('--vae_nvp_h_dim', type=int, default='200', help='Size of the hidden layers in each coupling layer in VAE_RealNVP')
    PARSER.add_argument('--vae_nvp_pretrained_model_dir', type=str, default='./mnist_vae_nvp/checkPoint/50.pt', help='The pretrained model dir of VAE_RealNVP')

    # Show images
    PARSER.add_argument('--if_show_images', type=bool, default=True, help='if show images')

    # Steps
    PARSER.add_argument('--steps', type=int, default=30, help='Steps to evaluate the average loss')


    HPARAMS = PARSER.parse_args()

    main(HPARAMS)