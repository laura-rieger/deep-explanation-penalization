#from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
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
sys.path.append('../fit')
from score_funcs import gradient_sum,eg_scores_2d
import cd
import torchvision.datasets as datasets
#XXX changed here without trying if fix
model_path = "../mnist_adversarial"

def save(p,  out_name):
    # save final
    
    os.makedirs(model_path, exist_ok=True)
    pkl.dump(s._dict(), open(os.path.join(model_path, out_name + '.pkl'), 'wb'))
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        
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
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--regularizer_rate', type=float, default=0.0, metavar='N',
                    help='how heavy to regularize lower order interaction (AKA color)')
parser.add_argument('--grad_method', type=int, default=0, metavar='N',
                    help='which gradient method is used - Grad or CD')
# parser.add_argument('--gradient_method', type=string, default="CD", metavar='N',
                    # help='what method is used')
args = parser.parse_args()
s = S(args.epochs)
use_cuda = not args.no_cuda and torch.cuda.is_available()
regularizer_rate = args.regularizer_rate
s.regularizer_rate = regularizer_rate
num_blobs = 32
s.num_blobs = num_blobs
s.seed = args.seed
#regularizer_rate /=num_blobs
#sys.exit()


device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


mnist_trainset = datasets.FashionMNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

num_train = int(len(mnist_trainset)*.9)
num_test = len(mnist_trainset)  - num_train 
torch.manual_seed(0);
train_dataset, test_dataset,= torch.utils.data.random_split(mnist_trainset, [num_train, num_test])
train_loader = utils.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs) # create your dataloader
test_loader = utils.DataLoader(test_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs) # create your dataloader


val_dataset = datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
val_loader = utils.DataLoader(val_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs) # create your dataloader


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


# make the sampling thing


blobs = np.zeros((28*28,28,28))
for i in range(28):
    for j in range(28):
        blobs[i*28+j, i, j] =1



prob = np.ones((28,28))

prob = prob.reshape(-1)
prob/=prob.sum()
model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum )

#optimizer = optim.Adam(model.parameters(), weight_decay = 0.0001) 

def train(args, model, device, train_loader, optimizer, epoch, regularizer_rate, until_batch = -1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if until_batch !=-1 and batch_idx > until_batch:
            break
        data, target = data.to(device), target.to(device)
         
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        
        if regularizer_rate !=0:
            add_loss = torch.zeros(1,).cuda()
            blob_idxs = np.random.choice(28*28, size = num_blobs, p = prob)
            if args.grad_method ==0:
                for i in range(num_blobs): 
                    rel, irrel = cd.cd(blobs[blob_idxs[i]], data,model)
                    add_loss += torch.nn.functional.softmax(torch.stack((rel.view(-1),irrel.view(-1)), dim =1), dim = 1)[:,0].mean()
                (regularizer_rate*add_loss+loss).backward()
            elif args.grad_method ==1:
                for i in range(num_blobs): 
                    add_loss +=gradient_sum(data, target, torch.FloatTensor(blobs[blob_idxs[i]]).to(device),  model, F.nll_loss)

                (regularizer_rate*add_loss).backward()
                loss = F.nll_loss(output, target)
                loss.backward()
            elif args.grad_method ==2:
                for j in range(len(data)):
                    for i in range(num_blobs): 
                        add_loss +=(eg_scores_2d(model, data, j, targets, num_samples) * torch.FloatTensor(blobs[blob_idxs[i]]).to(device)).sum()

                (regularizer_rate*add_loss).backward()
                loss = F.nll_loss(output, target)
                loss.backward()
                
                
 
            
        else:
            add_loss =torch.zeros(1,)
            loss.backward()
         
        optimizer.step()
        
        
        if batch_idx % args.log_interval == 0:
            pred = output.argmax(dim=1, keepdim=True)
            acc = 100.*pred.eq(target.view_as(pred)).sum().item()/len(target)
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Acc: ({:.0f}%), CD Loss: {:.6f}'.format(
                # epoch, batch_idx * len(data), len(train_loader.dataset),
                # 100. * batch_idx / len(train_loader), loss.item(),acc,   add_loss.item()))
              
            s.losses_train.append(loss.item())
            s.accs_train.append(acc)
            s.cd.append(add_loss.item())
   


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
    s.losses_test.append(test_loss)
    s.accs_test.append(100. * correct / len(test_loader.dataset))
    return test_loss


 

best_model_weights = None
best_test_loss = 100000
# train(args, model, device, train_loader, optimizer, 0, 0, until_batch = 3)
patience = 2
cur_patience = 0

for epoch in range(1, args.epochs + 1):

    train(args, model, device, train_loader, optimizer, epoch, regularizer_rate)
    test_loss = test(args, model, device, test_loader, epoch)
    if test_loss < best_test_loss:
        
        cur_patience = 0
        best_test_loss = test_loss
        best_model_weights = deepcopy(model.state_dict())
    else:
        cur_patience +=1
        if cur_patience > patience:
            break
 
s.dataset= "Fashion"      
print("FF")
test(args, model, device, val_loader, epoch+1)
if args.grad_method ==0:
    s.method = "CD"
elif args.grad_method ==2:
    s.method = "ExpectedGrad"
else:
    s.method = "Grad"
s.model_weights = best_model_weights
np.random.seed()
pid = ''.join(["%s" % np.random.randint(0, 9) for num in range(0, 20)])
save(s,  pid)
# if (args.save_model):

    # torch.save(model.state_dict(),oj(model_path, "color_mnist_cnn.pt"))