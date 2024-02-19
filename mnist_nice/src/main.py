import os
import torch
import torch.optim as optim
from torchvision import transforms, datasets

from nice import NICE
from argparse import ArgumentParser 
import utils

def main(hparams):

    mnist_train, mnist_test = utils.mnist_dataloader(hparams)
    model = NICE(hparams,data_dim=784)

    # dir = './mnist_nice/checkPoint/19.pt'
    # # checkpoint = torch.load(dir)
    # model.load_state_dict(torch.load(dir))

    if hparams.use_cuda == True:
        device = torch.device('cuda')
        model = model.to(device)

    model.train()
    opt = optim.Adam(model.parameters())

    # print(model)
    # print(model.scaling_layer.log_scale_vector.shape)
    # first_linear_layer_params = list(model.coupling_layers[0].m[0].parameters())
    # weight, bias = first_linear_layer_params
    # print(weight)
    # print(bias.shape)
    # print(list(model.coupling_layers[0].m[0].parameters()))


    for i in range(hparams.epochs):
        mean_likelihood = 0.0
        num_minibatches = 0

        for _, (x,_) in enumerate(mnist_train):
            x = x.view(-1, 784) + torch.rand(784) / 256.

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

        mean_likelihood /= num_minibatches
        print('Epoch {} completed. Log Likelihood: {}'.format(i, mean_likelihood))

        if not os.path.exists(hparams.save_dir):
            os.makedirs(hparams.save_dir)

        if (i+1) % 5 == 0:
            save_path = os.path.join(hparams.save_dir, '{}.pt'.format(i))
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Datasets
    PARSER.add_argument('--batch_size', type=int, default='256', help='The shape of mnist')

    # Adam
    PARSER.add_argument('--learning_rate', type=float, default='0.01', help='learning_rate of Adam')

    # Coupling Layer Parameters
    PARSER.add_argument('--h_dim', type=int, default='1000', help='The shape of mnist')
    PARSER.add_argument('--num_net_layers', type=int, default='6', help='The shape of mnist')

    # NICE parameters
    PARSER.add_argument('--num_coupling_layers', type=int, default='4', help='The shape of mnist')

    # Training
    PARSER.add_argument('--use_cuda', type=str, default='True', help='The shape of mnist')
    PARSER.add_argument('--epochs', type=int, default='60', help='The shape of mnist')
    PARSER.add_argument('--save_dir', type=str, default='./mnist_nice/checkPoint', help='The shape of mnist')
    PARSER.add_argument('--result_dir', type=str, default='./mnist_nice/NICEResult', help='Output directory')


    HPARAMS = PARSER.parse_args()
    main(HPARAMS)