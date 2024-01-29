import os
import numpy as np
import utils
import torch
from torch import optim
from torchvision.utils import save_image

from model_def import VAE
from argparse import ArgumentParser 

# Set device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def test(model, optimizer, mnist_test, epoch, best_test_loss, hparams):
    test_avg_loss = 0.0
    with torch.no_grad():  # 这一部分不计算梯度，也就是不放入计算图中去
        '''测试测试集中的数据'''
        # 计算所有batch的损失函数的和
        for test_batch_index, (test_x, _) in enumerate(mnist_test):
            test_x = test_x.to(device)
            # 前向传播
            test_x_hat, test_mu, test_log_var = model(test_x)
            # 损害函数值
            test_loss, test_BCE, test_KLD = utils.get_loss(test_x_hat, test_x, test_mu, test_log_var)
            test_avg_loss += test_loss

        # 对和求平均，得到每一张图片的平均损失
        test_avg_loss /= len(mnist_test.dataset)

        '''测试随机生成的隐变量'''

        z = torch.randn(hparams.batch_size, hparams.n_z).to(device)  

        random_res = model.decoder(z).view(-1, 1, 28, 28)

        save_image(random_res, './%s/random_sampled-%d.png' % (hparams.result_dir, epoch + 1))

        '''保存目前训练好的模型'''

        is_best = test_avg_loss < best_test_loss
        best_test_loss = min(test_avg_loss, best_test_loss)
        utils.save_checkpoint({
            'epoch': epoch,  # 迭代次数
            'best_test_loss': best_test_loss,  # 目前最佳的损失函数值
            'state_dict': model.state_dict(),  # 当前训练过的模型的参数
            'optimizer': optimizer.state_dict(),
        }, is_best, hparams.save_dir)

        return best_test_loss



def main(hparams):
    # Set up some stuff according to hparams
    utils.print_hparams(hparams)
    mnist_test, mnist_train = utils.mnist_dataloader(hparams)

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

    # Training
    loss_epoch = []
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
                print(f"Epoch [{epoch+1}/{hparams.training_epochs}], Batch [{batch_idx + 1}/{len(mnist_train.dataset)}]: \
                      Total-loss = {loss.item()/hparams.batch_size}, BCE-Loss = {BCE.item()/hparams.batch_size}, \
                        KLD-loss = {KLD.item()/hparams.batch_size}")

            if batch_idx == 0:
                # visualize reconstructed result at the beginning of each epoch
                x_concat = torch.cat([x_batch_val.view(-1, 1, 28, 28), x_hat.view(-1, 1, 28, 28)], dim=3)
                save_image(x_concat, './%s/reconstructed-%d.png' % (hparams.result_dir, epoch + 1))

        # 把这一个epoch的每一个样本的平均损失存起来
        loss_epoch.append(np.sum(loss_batch) / len(mnist_train.dataset))  # len(mnist_train.dataset)为样本个数

        # 测试模型
        if (epoch + 1) % hparams.test_every == 0:
            best_test_loss = test(model, optimizer, mnist_test, epoch, best_test_loss, hparams)
    return loss_epoch



if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Base parameters in VAE
    PARSER.add_argument('--n_input', type=int, default='784', help='The shape of MNIST tensor')
    PARSER.add_argument('--n_z', type=int, default='20', help='The shape of latent variable z')
    PARSER.add_argument('--h_dim', type=int, default='500', help='The dim of hidden')
    PARSER.add_argument('--batch_size', type=int, default='128', help='The dim of hidden')


    # Training Parameters
    PARSER.add_argument('--learning_rate', type=float, default='1e-3', help='The learning rate of optimize')
    PARSER.add_argument('--training_epochs', type=int, default='100', help='The training epochs of optimize')

    # Testing Parameters
    PARSER.add_argument('--test_every', type=int, default='10', help='Test after every epochs')

    # Checkpoint dir
    PARSER.add_argument('--resume', type=str, default='None', help='Path to latest checkpoint')
    PARSER.add_argument('--result_dir', type=str, default='./VAEResult', help='Output directory')
    PARSER.add_argument('--save_dir', type=str, default='./checkPoint', help='Model saving directory')
    

    # HPARAMS.num_samples = 60000
    # HPARAMS.learning_rate = 0.001
    # HPARAMS.batch_size = 100
    # HPARAMS.training_epochs = 100
    # HPARAMS.summary_epoch = 1
    # HPARAMS.ckpt_epoch = 5

    HPARAMS = PARSER.parse_args()
    main(HPARAMS)
