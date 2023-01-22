from torchvision import datasets

datasets.MNIST('./data', train=True, download=True)
datasets.FashionMNIST('./data', train=True, download=True)