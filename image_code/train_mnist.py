#from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import join as oj
import torch.utils.data as utils
from torchvision import datasets, transforms
import numpy as np
import os
import sys
import pickle as pkl
from copy import deepcopy
from params_save import S # class to save objects
sys.path.append('../../acd//acd/scores')
import cd
model_path = "../img_models"

def save(p,  out_name):
    # save final
    os.makedirs(model_path, exist_ok=True)
    pkl.dump(s._dict(), open(os.path.join(model_path, out_name + '.pkl'), 'wb'))
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    def logits(self, x):
    
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return x

    



# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--regularizer_rate', type=float, default=0.0, metavar='N',
                    help='how heavy to regularize lower order interaction (AKA color)')

args = parser.parse_args()
s = S(args.epochs)
use_cuda = not args.no_cuda and torch.cuda.is_available()
regularizer_rate = args.regularizer_rate
s.regularizer_rate = regularizer_rate
num_blobs = 32
s.num_blobs = num_blobs

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_x_tensor = torch.Tensor(np.load(oj("../data/ColorMNIST", "train_x.npy")))
train_y_tensor = torch.Tensor(np.load(oj("../data/ColorMNIST", "train_y.npy"))).type(torch.int64)
train_dataset = utils.TensorDataset(train_x_tensor,train_y_tensor) # create your datset
train_loader = utils.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs) # create your dataloader

test_x_tensor = torch.Tensor(np.load(oj("../data/ColorMNIST", "test_x.npy")))
test_y_tensor = torch.Tensor(np.load(oj("../data/ColorMNIST", "test_y.npy"))).type(torch.int64)
test_dataset = utils.TensorDataset(test_x_tensor,test_y_tensor) # create your datset
test_loader = utils.DataLoader(test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs) # create your dataloader


# make the sampling thing

blobs = np.zeros((28*28,28,28))
for i in range(28):
    for j in range(28):
        blobs[i*28+j, i, j] =1


prob = ((train_x_tensor).numpy().sum(axis = 1) !=-3).mean(axis = 0).reshape(-1)
prob /=prob.sum()
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)



def train(args, model, device, train_loader, optimizer, epoch, regularizer_rate):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
         
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        
        if regularizer_rate !=0:
            add_loss = torch.zeros(1,).cuda()
            blob_idxs = np.random.choice(28*28, size = num_blobs, p = prob)

            for i in range(num_blobs): 
                rel, irrel = cd.cd(blobs[blob_idxs[i]], data,model)
                if (not torch.isnan(rel).any()) and (not torch.isnan(irrel).any()):
                    add_loss += torch.nn.functional.softmax(torch.stack((rel.view(-1),irrel.view(-1)), dim =1), dim = 1)[:,0].mean()

       
   
            (regularizer_rate*add_loss+loss).backward()
        else:
            add_loss =torch.zeros(1,)
            loss.backward()
         
        optimizer.step()
        
        
        if batch_idx % args.log_interval == 0:
            pred = output.argmax(dim=1, keepdim=True)
            acc = 100.*pred.eq(target.view_as(pred)).sum().item()/len(target)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Acc: ({:.0f}%), CD Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),acc,   add_loss.item()))
              
            s.losses_train[epoch-1] = loss.item()
            s.accs_train[epoch-1] = acc
            s.cd[epoch-1] = add_loss.item()


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    s.losses_test[epoch-1] = test_loss
    s.accs_test[epoch-1] = 100. * correct / len(test_loader.dataset)





for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch, regularizer_rate)
    test(args, model, device, test_loader, epoch)
s.model_weights = deepcopy(model.state_dict())
pid = ''.join(["%s" % np.random.randint(0, 9) for num in range(0, 20)])
save(s,  pid+ ".pkl")
# if (args.save_model):

    # torch.save(model.state_dict(),oj(model_path, "color_mnist_cnn.pt"))
    
