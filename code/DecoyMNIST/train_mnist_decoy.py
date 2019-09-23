
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
import time
sys.path.append('../.')
import cd
from score_funcs import gradient_sum, eg_scores_2d
def save(p,  out_name):
    # save final
    os.makedirs(model_path, exist_ok=True)
    pkl.dump(s._dict(), open(os.path.join(model_path, out_name + '.pkl'), 'wb'))
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 256)
        self.fc2 = nn.Linear(256, 10)

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
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
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
                    help='how heavy to regularize lower order interaction (AKA color)')
# parser.add_argument('--gradient_method', type=string, default="CD", metavar='N',
                    # help='what method is used')
args = parser.parse_args()
model_path = "../../models/DecoyMNIST"
s = S(args.epochs)
use_cuda = not args.no_cuda and torch.cuda.is_available()
regularizer_rate = args.regularizer_rate
s.regularizer_rate = regularizer_rate
num_blobs = 1
num_samples =200
s.num_blobs = num_blobs
s.seed = args.seed

#sys.exit()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_x_tensor = torch.Tensor(np.load(oj("../../data/ColorMNIST", "train_x_decoy.npy")))
train_y_tensor = torch.Tensor(np.load(oj("../../data/ColorMNIST", "train_y.npy"))).type(torch.int64)
complete_dataset = utils.TensorDataset(train_x_tensor,train_y_tensor) # create your datset


num_train = int(len(complete_dataset)*.9)
num_test = len(complete_dataset)  - num_train 
torch.manual_seed(0);
train_dataset, test_dataset,= torch.utils.data.random_split(complete_dataset, [num_train, num_test])
train_loader = utils.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs) # create your dataloader
test_loader = utils.DataLoader(test_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs) # create your dataloader

test_x_tensor = torch.Tensor(np.load(oj("../../data/ColorMNIST", "test_x_decoy.npy")))
test_y_tensor = torch.Tensor(np.load(oj("../../data/ColorMNIST", "test_y.npy"))).type(torch.int64)
val_dataset = utils.TensorDataset(test_x_tensor,test_y_tensor) # create your datset
val_loader = utils.DataLoader(val_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs) # create your dataloader


# make the sampling thing



blob = np.zeros((28,28))
size_blob =5
blob[:size_blob, :size_blob ] =1

blob[-size_blob:, :size_blob] = 1
blob[:size_blob, -size_blob: ] =1
blob[-size_blob:, -size_blob:] = 1



model = Net().to(device)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

optimizer = optim.Adam(model.parameters(),weight_decay = 0.0001) #, lr=args.lr, momentum=args.momentum)

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

            
            if args.grad_method ==0:
                rel, irrel = cd.cd(blob, data,model)
                add_loss += torch.nn.functional.softmax(torch.stack((rel.view(-1),irrel.view(-1)), dim =1), dim = 1)[:,0].mean()

                #print(torch.cuda.max_memory_allocated(0)/np.power(10,9))
                (regularizer_rate*add_loss +loss).backward()
            elif args.grad_method ==1:
                add_loss +=gradient_sum(data, target, torch.FloatTensor(blob).to(device),  model, F.nll_loss)
                (regularizer_rate*add_loss).backward()

                #print(torch.cuda.max_memory_allocated(0)/np.power(10,9))
                optimizer.step()
                loss = F.nll_loss(output, target)
                loss.backward()

            elif args.grad_method ==2:
                for j in range(len(data)):
                    add_loss +=(eg_scores_2d(model, data, j, target, num_samples) * torch.FloatTensor(blob).to(device)).sum()
                (regularizer_rate*add_loss).backward()

                #print(torch.cuda.max_memory_allocated(0)/np.power(10,9))
                optimizer.step()
                loss = F.nll_loss(output, target)
     
                loss.backward()
        else:
            add_loss = torch.zeros(1,)
            loss.backward()

        #print(torch.cuda.max_memory_allocated(0)/np.power(10,9))
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


start = time.time()

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

end = time.time()
s.time_per_epoch = (end - start)/(epoch)

        
s.model_weights = best_model_weights
print("FF")
test(args, model, device, val_loader, epoch+1)
s.dataset= "Decoy"
if args.grad_method ==0:
    s.method = "CDEP"
elif args.grad_method ==2:
    s.method = "ExpectedGrad"
else:
    s.method = "Grad"
np.random.seed()
pid = ''.join(["%s" % np.random.randint(0, 9) for num in range(0, 20)])
save(s,  pid)
