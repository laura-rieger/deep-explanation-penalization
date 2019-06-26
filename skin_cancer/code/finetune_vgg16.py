import torch
import torchvision
import torchvision.datasets as datasets
import sys
import numpy as np
import torch.utils.data as utils
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle as pkl
from os.path import join as oj
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import torch
import torchvision
import argparse
import torchvision.datasets as datasets
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as utils
from torch import nn
from numpy.random import randint
import torchvision.models as models
import time
import os
import copy
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--last_layer_only', action='store_false', default=True,
                    help='disables CUDA training')

args = parser.parse_args()



feature_extract = args.last_layer_only # only 
num_epochs = args.epochs

device = torch.device(0)

# load model
model = models.vgg16(pretrained=True)
# make conv untrainable - test if needed
model.classifier[-1] = nn.Linear(4096, 2)
model = model.to(device)
if feature_extract:
    for param in model.features.parameters():
        param.requires_grad = False


# load data
data_root  = "../../../datasets/ISIC"
data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
img_dataset = datasets.ImageFolder(data_root,transform=data_transform)
num_total = len(img_dataset)
num_train = int(0.8 * num_total)
num_val = int(0.1 * num_total)
num_test = num_total - num_train - num_val
torch.manual_seed(0);
train_dataset, test_dataset, val_dataset= torch.utils.data.random_split(img_dataset, [num_train, num_test, num_val])


# get filtered indices
img_list =img_dataset.imgs
file_names = [x[0] for x in img_list]
start_patch_ind = file_names.index('../../../datasets/ISIC/not_cancer/ISIC_0000557.jpg')
end_patch_ind = file_names.index('../../../datasets/ISIC/not_cancer/ISIC_0009867.jpg')
patch_indice = np.arange(start_patch_ind,end_patch_ind+1)
start_ruler_ind = file_names.index('../../../datasets/ISIC/cancer/ISIC_0012099.jpg')
end_ruler_ind = file_names.index('../../../datasets/ISIC/cancer/ISIC_0014559.jpg')
ruler_indice = np.arange(start_ruler_ind,end_ruler_ind+1)
filter_idx = np.concatenate((patch_indice, ruler_indice))
train_filtered = Subset(img_dataset, list(filter(lambda x: x in filter_idx, train_dataset.indices)) )

datasets = {'train' : train_filtered, 'test':test_dataset, 'val': val_dataset}
dataset_sizes = {'train' : len(train_filtered), 'test':len(test_dataset), 'val': len(val_dataset)}
torch.manual.seed(args.seed)

not_cancer_ratio = np.asarray(train_filtered.dataset.targets).mean() 
cancer_ratio = 1- not_cancer_ratio
cancer_weight = 1/cancer_ratio
not_cancer_weight = 1/ not_cancer_ratio
weights = np.asarray([cancer_weight, not_cancer_weight])
weights /= weights.sum()
weights = torch.tensor(weights).to(device)


dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test','val']}


def train_model(model,dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase])):
    
                inputs = inputs.to(device)
                labels = labels.to(device)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'val' and epoch_acc > best_acc:
            
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,val_acc_history #TODO hist
    
    

params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:


    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            pass
            # print("\t",name)
            
            
criterion = nn.CrossEntropyLoss(weight = weights.double().float())

#sys.exit()
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

model, hist = train_model(model, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)
pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])
torch.save(model.state_dict(),oj("../models", pid + ".pt"))
import pickle as pkl
pkl.dump(hist, open(os.path.join('../models' , pid + '.pkl'), 'wb'))