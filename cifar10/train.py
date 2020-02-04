import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append('../src')
import score_funcs
import argparse
import numpy as np
from score_funcs import gradient_sum,eg_scores_2d,cdep
import os
import random
from tqdm import tqdm
import torch.nn as nn
import pickle as pkl
from copy import deepcopy
from params import S
from model import Net 


model_path = "../models/cifar10_equal_loss"
torch.backends.cudnn.deterministic = True #this makes results reproducible. 
def save(p,  out_name):
    # save final
    os.makedirs(model_path, exist_ok=True)
    pkl.dump(p._dict(), open(os.path.join(model_path, out_name + '.pkl'), 'wb'))
    

    



# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--regularizer_rate', type=float, default=1.0, metavar='N',
                    help='how heavy to regularize lower order interaction (AKA color)')
parser.add_argument('--grad_method', type=int, default=0, metavar='N',
                    help='which gradient method is used - Grad or CD')

args = parser.parse_args()


num_epochs = args.epochs 
batch_size = args.batch_size
regularizer_rate = args.regularizer_rate
test_batch_size = args.test_batch_size
s = S(num_epochs)
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

s.regularizer_rate = regularizer_rate
num_blobs = 4
s.num_blobs = num_blobs
s.seed = args.seed

img_width = 32
blob_size = 1
num_blobs_width = int(img_width/blob_size)
blobs = np.zeros((num_blobs_width*num_blobs_width,img_width,img_width))
for i in range(num_blobs_width):
    for j in range(num_blobs_width):
        blobs[i*num_blobs_width+j, i*blob_size:(i+1)*blob_size, j*blob_size:(j+1)*blob_size] =1
prob =np.ones((len(blobs)))
prob /=prob.sum()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
                                        
                                        
                                        
torch.manual_seed(0);
num_train = int(0.9*len(dataset))
num_val = len(dataset) - num_train
train_dataset, val_dataset= torch.utils.data.random_split(dataset, [num_train, num_val])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           

torch.manual_seed(args.seed) #weight init is varied
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
net = Net().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())#,lr=0.01, momentum=0.9 )
log_interval = 100
def train(model, device, train_loader, optimizer, epoch, regularizer_rate = 0):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
         
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        add_loss = torch.zeros(1,).cuda()
        if regularizer_rate !=0:
            
            blob_idxs = np.random.choice(num_blobs_width*num_blobs_width, size = num_blobs, p = prob)
            for i in range(num_blobs): 
                add_loss += score_funcs.cdep(model, data, blobs[blob_idxs[i]], )

        (regularizer_rate*add_loss+loss).backward()
       
         
        optimizer.step()
        
        
        if batch_idx % log_interval == 0:
            pred = output.argmax(dim=1, keepdim=True)
            acc = 100.*pred.eq(target.view_as(pred)).sum().item()/len(target)
            s.losses_train.append(loss.item())
            s.accs_train.append(acc)
            s.cd.append(add_loss.item())
            #print(add_loss)
   

best_model_weights = None
best_test_loss = 100000
patience = 2
cur_patience = 0

def test( model, device, data_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print('{} \nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    s.losses_test.append(test_loss)
    s.accs_test.append(100. * correct / len(data_loader.dataset))
    return test_loss





for epoch in range(1, num_epochs + 1):

    train( net, device, train_loader, optimizer, epoch, regularizer_rate = regularizer_rate)
    test_loss = test( net, device, val_loader, epoch)
    if test_loss < best_test_loss:
        
        cur_patience = 0
        best_test_loss = test_loss
        best_model_weights = deepcopy(net.state_dict())
    else:
        cur_patience +=1
        if cur_patience > patience:
            break
 
net.load_state_dict(best_model_weights)
test(net, device, val_loader, epoch+1)
if args.grad_method ==0:
    s.method = "CDEP"

s.model_weights = best_model_weights
s.batch_size = args.batch_size
s.regularizer_rate = args.regularizer_rate
print(s.seed)
np.random.seed()
pid = ''.join(["%s" % np.random.randint(0, 9) for num in range(0, 20)])
save(s,  pid)
