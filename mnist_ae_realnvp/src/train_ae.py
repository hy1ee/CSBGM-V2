import torch
from torch import nn
import utils
from argparse import ArgumentParser 
import itertools
import os
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    """
    A simple autoencoder for images. 
    self.linear1 generates the intermediate embeddings that we use for the normalizing flow.
    """
    def __init__(self,hparams):
        super().__init__()
        
        self.hparams = hparams
        # Encoding layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, stride=2, kernel_size=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=3, bias=False, padding=1)
        self.linear1 = nn.Linear(in_features=3136, out_features=hparams.embedding_dim)
        
        # Decoding layers
        self.linear2 = nn.Linear(in_features=hparams.embedding_dim, out_features=3136)
        self.convt1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, stride=2, kernel_size=3, padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=32, out_channels=1, stride=2, kernel_size=3, padding=1, output_padding=1)


    def forward(self, x):
        
        emb = self.encoder(x)
        _x = self.decoder(emb)
        
        return _x, emb
    
    def decoder(self, emb):

        _x = torch.relu(self.linear2(emb))
        _x = _x.view(-1, 64, 7, 7)
        _x = torch.relu(self.convt1(_x))
        _x = self.convt2(_x)
        
        return _x
    
    def encoder(self, x):
        _x = torch.relu(self.conv1(x))
        _x = torch.relu(self.conv2(_x))
        sh = _x.shape

        _x = torch.relu(torch.flatten(_x, 1))
        
        emb = self.linear1(_x)
        
        return emb


def main(hparams):
    if hparams.use_cuda:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = "cpu"

    mnist_train, mnist_test = utils.mnist_dataloader(hparams)

    loss_f = nn.BCELoss()

    # Build the autoencoder
    model = AutoEncoder(hparams)
    model = model.to(device)

    optimizer = torch.optim.Adam(itertools.chain(model.parameters()),lr=hparams.learning_rate, weight_decay=hparams.weight_decay)


    if not os.path.exists(hparams.result_dir):
        os.makedirs(hparams.result_dir)

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    Total_losses = []
    for epoch in range(hparams.training_epochs):
        loss_epoch = 0.0
        num_minibatches = 0
        
        for _, data in enumerate(mnist_train):

            x, _ = data
            x = x.to(device)

            _x, emb = model(x)
            _x = torch.sigmoid(_x)
            rec_loss = loss_f(_x, x)

            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()    

            loss_epoch += rec_loss
            num_minibatches += 1

        loss_epoch /= num_minibatches
        print('Epoch {} completed. Log Likelihood: {}'.format(epoch, loss_epoch))
        Total_losses.append(loss_epoch)
    
        if (epoch+1) % 10 == 0:
            save_path = os.path.join(hparams.save_dir, 'AE_{}.pt'.format(epoch+1))
            torch.save(model.state_dict(), save_path)

    total_loss = [loss.detach().cpu().numpy() for loss in Total_losses]
    x_plt = range(len(total_loss))

    plt.plot(x_plt, total_loss, label='Total Loss')

    plt.title('AE Loss') 
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./%s/AE_Training_loss.png' % (hparams.result_dir))

        
if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Datasets
    PARSER.add_argument('--batch_size', type=int, default='100', help='Batch size for each epoch')

    # Adam
    PARSER.add_argument('--learning_rate', type=float, default='1e-4', help='learning_rate of Adam')
    PARSER.add_argument('--weight_decay', type=float, default='1e-5', help='Weight_decay of Adam')

    # AE parameters
    PARSER.add_argument('--embedding_dim', type=int, default='20', help='The output dim of AE')

    # Training
    PARSER.add_argument('--use_cuda', type=bool, default=True, help='Device')
    PARSER.add_argument('--training_epochs', type=int, default='50', help='The Training epochs')
    PARSER.add_argument('--save_dir', type=str, default='./mnist_ae_realnvp/checkPoint', help='Path to store model parameters')
    PARSER.add_argument('--result_dir', type=str, default='./mnist_ae_realnvp/AERealNVP_Result', help='Path to store model results')


    HPARAMS = PARSER.parse_args()
    main(HPARAMS)