import os
import torch
import torch.optim as optim

from nice import NICE
from argparse import ArgumentParser 
import utils
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def main(hparams):

    mnist_train, mnist_test = utils.mnist_dataloader(hparams)
    model = NICE(hparams, data_dim=hparams.n_input)

    if hparams.use_cuda == True:
        device = torch.device('cuda')
        model = model.to(device)

    model.train()
    opt = optim.Adam(model.parameters())

    if not os.path.exists(hparams.result_dir):
        os.makedirs(hparams.result_dir)


    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    Total_losses = []
    for epoch in range(hparams.epochs):
        mean_likelihood = 0.0
        num_minibatches = 0

        for batch_idx, (x,_) in enumerate(mnist_train):
            x_batch_val = x
            x = x.view(-1, hparams.n_input) + torch.rand(hparams.n_input) / 256.

            if hparams.use_cuda == True:
                x = x.cuda()
            
            x = torch.clamp(x, 0, 1)
            z, likelihood = model(x)
            loss = -torch.mean(likelihood)   # NLL

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            mean_likelihood -= loss
            num_minibatches += 1

            if batch_idx == 0:
                x_hat =  model(z, invert=True)
                # visualize reconstructed result at the beginning of each epoch
                x_concat = torch.cat([x_batch_val.view(-1, 1, 28, 28), x_hat.view(-1, 1, 28, 28)], dim=3)
                save_image(x_concat, './%s/reconstructed-%d.png' % (hparams.result_dir, epoch + 1))


        mean_likelihood /= num_minibatches
        print('Epoch {} completed. Log Likelihood: {}'.format(epoch, mean_likelihood))
        Total_losses.append(mean_likelihood)

        if (epoch+1) % 5 == 0:
            save_path = os.path.join(hparams.save_dir, '{}.pt'.format(epoch+1))
            torch.save(model.state_dict(), save_path)

            z_random_sample = torch.randn(hparams.batch_size, hparams.n_input)# .to(device)  
            random_res = model(z_random_sample, invert=True).view(-1, 1, 28, 28)

            save_image(random_res, './%s/random_sampled-%d.png' % (hparams.result_dir, epoch + 1))

    x_plt = range(len(Total_losses))

    plt.plot(x_plt, Total_losses, label='Total Loss')

    plt.title('NICE Loss')
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./%s/loss.png' % (hparams.result_dir))

        
if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Datasets
    PARSER.add_argument('--n_input', type=int, default='784', help='The dim of input')
    PARSER.add_argument('--batch_size', type=int, default='100', help='Batch size for each epoch')

    # Adam
    PARSER.add_argument('--learning_rate', type=float, default='0.01', help='learning_rate of Adam')

    # NICE Parameters
    PARSER.add_argument('--nice_h_dim', type=int, default='1000', help='The hidden linear dim of Layer in NICE')
    PARSER.add_argument('--num_net_layers', type=int, default='6', help='The num of net layers in NICE')
    PARSER.add_argument('--num_coupling_layers', type=int, default='4', help='The num of coupling layers in NICE')

    # Training
    PARSER.add_argument('--use_cuda', type=str, default='True', help='Device')
    PARSER.add_argument('--epochs', type=int, default='100', help='The Training epochs')
    PARSER.add_argument('--save_dir', type=str, default='./mnist_nice/checkPoint', help='Path to store model parameters')
    PARSER.add_argument('--result_dir', type=str, default='./mnist_nice/NICEResult', help='Path to store model results')


    HPARAMS = PARSER.parse_args()
    main(HPARAMS)