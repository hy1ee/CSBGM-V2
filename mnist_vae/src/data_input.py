import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def mnist_data_iterator(hparams, num_batches):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download MNIST dataset
    test_dataset = train_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hparams.batch_size, shuffle=True)

    def iterator():
        for _ in range(num_batches):
            for data, target in train_loader:
                # Assuming data is a batch of images and target is a batch of labels
                # You might need to one-hot encode the labels if necessary
                yield data, target

    return iterator
