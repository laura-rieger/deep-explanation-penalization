import torch
import torchvision
import torchvision.datasets as datasets
import sys
import numpy as np
import torch.utils.data as utils
from tqdm import tqdm
from colour import Color
red = Color("red")
colors = list(red.range_to(Color("purple"),10))
colors = [x.get_rgb() for x in colors]
from os.path import join as oj
mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=None)
num_samples = len(mnist_trainset)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)

noise = 0.0
np.random.seed(0)
for i in tqdm(range(num_samples)):
    my_color  = np.clip(colors[mnist_trainset.targets[i]]+ np.random.normal(size=3,scale =noise),0,1)
    color_x[i ] = mnist_trainset.data[i].numpy().astype(np.float32)[np.newaxis]*my_color[:, None, None]
color_y = mnist_trainset.train_labels.numpy().copy()

color_x /=color_x.max()
color_x = color_x*2 -1
np.save(oj("../data/ColorMNIST", "train_x.npy"), color_x)
np.save(oj("../data/ColorMNIST", "train_y.npy"), color_y)


mnist_trainset = datasets.MNIST(root='../data', train=False, download=True, transform=None)
num_samples = len(mnist_trainset)
color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)
for i in tqdm(range(num_samples)):
    my_color  = np.clip(colors[9-mnist_trainset.targets[i]] + np.random.normal(size=3,scale =noise),0,1)
    color_x[i ] = mnist_trainset.data[i].numpy().astype(np.float32)[np.newaxis]*my_color[:, None, None]
color_y = mnist_trainset.train_labels.numpy().copy()

color_x /=color_x.max()
color_x = color_x*2 -1
np.save(oj("../data/ColorMNIST", "test_x.npy"),  color_x)
np.save(oj("../data/ColorMNIST", "test_y.npy"), color_y)
#np.save(oj("../data/ColorMNIST", "test_y_color.npy"), 9-color_y)
print("Saved color MNIST")