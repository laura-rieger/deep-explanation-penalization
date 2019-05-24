import torch
import torchvision
import torchvision.datasets as datasets
import matplotliblib.pyplot as plt
import numpy as np
import torch.utils.data as utils
from os.path import join as oj
mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=None)
color_x = np.zeros((60000, 3, 28, 28))
color_x[np.where((mnist_trainset.train_labels <5)), 0 ] =  mnist_trainset.data[np.where((mnist_trainset.train_labels <5))]
color_x[np.where((mnist_trainset.train_labels >=5)),1] =  mnist_trainset.data[np.where((mnist_trainset.train_labels >=5))]
color_y = mnist_trainset.train_labels.numpy().copy()
np.save(oj("../data/ColorMNIST", "train_x.npy"), color_x)
np.save(oj("../data/ColorMNIST", "train_y.npy"), color_y)
mnist_trainset = datasets.MNIST(root='../data', train=False, download=True, transform=None)
color_x = np.zeros((10000, 3, 28, 28))
color_x[np.where((mnist_trainset.train_labels >=5)), 0 ] =  mnist_trainset.data[np.where((mnist_trainset.train_labels <5))]
color_x[np.where((mnist_trainset.train_labels <5)),1] =  mnist_trainset.data[np.where((mnist_trainset.train_labels >=5))]
color_y = mnist_trainset.train_labels.numpy().copy()
np.save(oj("../data/ColorMNIST", "test_x.npy"), color_x)
np.save(oj("../data/ColorMNIST", "test_y.npy"), color_y)