import torch
import torchvision
import torchvision.datasets as datasets
import sys
import numpy as np
import torch.utils.data as utils
from colour import Color
red = Color("red")
colors = list(red.range_to(Color("blue"),10))
colors = [x.get_rgb() for x in colors]
from os.path import join as oj
mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=None)
color_x = np.zeros((60000, 3, 28, 28))


for i in range(10):
    color_x[np.where((mnist_trainset.train_labels ==i)) ] = (mnist_trainset.data[np.where((mnist_trainset.train_labels ==i))].numpy().astype(np.float32)[:, np.newaxis, :,:]*np.asarray(colors[i])[None, :, None, None])




color_y = mnist_trainset.train_labels.numpy().copy()

color_x /=color_x.max()
color_x = color_x*2 -1
np.save(oj("../data/ColorMNIST", "train_x.npy"), color_x)
np.save(oj("../data/ColorMNIST", "train_y.npy"), color_y)


mnist_trainset = datasets.MNIST(root='../data', train=False, download=True, transform=None)
color_x = np.zeros((10000, 3, 28, 28))
for i in range(10):
    color_x[np.where((mnist_trainset.train_labels ==i)) ] = (mnist_trainset.data[np.where((mnist_trainset.train_labels ==i))].numpy().astype(np.float32)[:, np.newaxis, :,:]*np.asarray(colors[9-i])[None, :, None, None])
color_y = mnist_trainset.train_labels.numpy().copy()
color_x /=color_x.max()
color_x = color_x*2 -1
np.save(oj("../data/ColorMNIST", "test_x.npy"),  color_x)
np.save(oj("../data/ColorMNIST", "test_y.npy"), color_y)
np.save(oj("../data/ColorMNIST", "test_y_color.npy"), 9-color_y)
print("Saved color MNIST")