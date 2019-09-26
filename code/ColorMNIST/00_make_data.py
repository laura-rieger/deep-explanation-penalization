import torch
import torchvision
import torchvision.datasets as datasets
import sys
import numpy as np
import torch.utils.data as utils
from tqdm import tqdm
from colour import Color
import os
from os.path import join as oj
np.random.seed(0)
red = Color("red")
colors = list(red.range_to(Color("purple"),10))
colors = [np.asarray(x.get_rgb()) for x in colors]


mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
num_samples = len(mnist_trainset)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)


for i in tqdm(range(num_samples)):
    my_color  = colors[mnist_trainset.train_labels[i].item()]
    color_x[i ] = mnist_trainset.data[i].numpy().astype(np.float32)[np.newaxis]*my_color[:, None, None]
color_y = mnist_trainset.train_labels.numpy().copy()
os.makedirs("../../data/ColorMNIST", exist_ok = True)

np.save(oj("../../data/ColorMNIST", "train_x.npy"), color_x)
np.save(oj("../../data/ColorMNIST", "train_y.npy"), color_y)


mnist_trainset = datasets.MNIST(root='../../data', train=False, download=True, transform=None)
num_samples = len(mnist_trainset)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)

color_y = mnist_trainset.train_labels.numpy().copy()
for i in tqdm(range(num_samples)):
    color_x[i ] = mnist_trainset.data[i].numpy().astype(np.float32)[np.newaxis]*colors[9-color_y[i]] [:, None, None]

np.save(oj("../../data/ColorMNIST", "test_x.npy"),  color_x)
np.save(oj("../../data/ColorMNIST", "test_y.npy"), color_y)
