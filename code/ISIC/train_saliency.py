import torch
import torchvision
import torchvision.datasets as datasets
import sys
import numpy as np
import torch.utils.data as utils
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import  transforms
import pickle as pkl
from os.path import join as oj

import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from torch.utils.data import TensorDataset, ConcatDataset, Subset
import argparse
import torchvision.datasets as datasets
import sys
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as utils
from torch import nn
from numpy.random import randint
import torchvision.models as models
import time
import os
import copy
sys.path.append('../')
import cd
import gc
from score_funcs import ig_scores_2d, gradient_sum
import json
with open('config.json') as json_file:
    data = json.load(json_file)
model_path = os.path.join(data["model_folder"], "feature_models_gradient")
data_path =data["data_folder"]

seg_path  = oj(data_path, "segmentation")
not_cancer_path = oj(data_path, "processed/benign")
cancer_path = oj(data_path, "processed/malignant")
 
 
mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--regularizer_rate', type=float, default=0.1, metavar='N',
                    help='how heavy to regularize lower order interaction (AKA color)')
args = parser.parse_args()

regularizer_rate = args.regularizer_rate

num_epochs = args.epochs

device = torch.device(0)

# load model
model = models.vgg16(pretrained=True)
# make conv untrainable - test if needed

model.classifier[-1] = nn.Linear(4096, 2)
model = model.to(device)


def load_folder(path):
    list_files= os.listdir(path)
    num_files = len(list_files)
    imgs_np = np.empty((num_files,  299, 299,3))
    for i in tqdm(range(num_files)):
        try:
            img = Image.open(oj(path, list_files[i]))
            imgs_np[i] = np.asarray(img)/255.0
            
            img.close()
        except:
            print(i)
    return imgs_np

def load_seg(path, orig_path):
    list_files= os.listdir(orig_path)
    num_files = len(list_files)
    imgs_np = np.zeros((num_files,  299, 299), dtype = np.bool)
    for i in tqdm(range(num_files)):
        if os.path.isfile(oj(path,  list_files[i])):
            img = Image.open(oj(path, list_files[i]))
            imgs_np[i] = np.asarray(img)[:,:,0] > 100
            img.close()
    return imgs_np

cancer_set = load_folder(cancer_path)
cancer_set -= mean[None, None, :]
cancer_set /= std[None, None, :]
not_cancer_set = load_folder(not_cancer_path)
not_cancer_set -= mean[None, None, :]
not_cancer_set /= std[None, None, :]
seg_set = load_seg(seg_path, not_cancer_path)

cancer_targets = np.ones((cancer_set.shape[0])).astype(np.int64)
not_cancer_targets = np.zeros((not_cancer_set.shape[0])).astype(np.int64)
not_cancer_dataset = TensorDataset(torch.from_numpy(not_cancer_set.swapaxes(1,3).swapaxes(2,3)).float(), torch.from_numpy(not_cancer_targets),torch.from_numpy(seg_set))
del not_cancer_set
del seg_set

cancer_dataset = TensorDataset(torch.from_numpy(cancer_set.swapaxes(1,3).swapaxes(2,2)).float(), torch.from_numpy(cancer_targets),torch.from_numpy(np.zeros((len(cancer_set), 299, 299), dtype = np.bool)))
del cancer_set


gc.collect()
complete_dataset = ConcatDataset((cancer_dataset, not_cancer_dataset))
num_total = len(complete_dataset)
num_train = int(0.8 * num_total)
num_val = int(0.1 * num_total)
num_test = num_total - num_train - num_val
torch.manual_seed(0);
train_dataset, test_dataset, val_dataset= torch.utils.data.random_split(complete_dataset, [num_train, num_test, num_val])
datasets = {'train' : train_dataset, 'test':test_dataset, 'val': val_dataset}
dataset_sizes = {'train' : len(train_dataset), 'test':len(test_dataset), 'val': len(val_dataset)}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test','val']}



       

cancer_ratio =len(cancer_dataset)/len(complete_dataset)


not_cancer_ratio = 1- cancer_ratio
cancer_weight = 1/cancer_ratio
not_cancer_weight = 1/ not_cancer_ratio
weights = np.asarray([not_cancer_weight, cancer_weight])
weights /= weights.sum()
weights = torch.tensor(weights).to(device)




 
def train_model(model,dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    val_acc_history = []
    val_loss_history = []
    train_loss_history = []
    
    train_acc_history = []
    train_cd_history= []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10.0
    


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
            running_loss_cd = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels, seg) in tqdm(enumerate(dataloaders[phase])):
    
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                seg = seg.to(device)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # need to do calc beforehand because we do need the gradients
                    if phase == 'train' and regularizer_rate !=0:
                        inputs.requires_grad = True
                        add_loss = gradient_sum(inputs, labels, seg, model, criterion)  
                        if add_loss!=0:
                            (regularizer_rate*add_loss).backward()
                            optimizer.step()
                        #print(torch.cuda.memory_allocated()/(np.power(10,9)))
                        optimizer.zero_grad()   
                        running_loss_cd +=add_loss.item() * inputs.size(0)
     
                        #inputs.require_grad = False
                   
                    
                         
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        (loss).backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_cd_loss = running_loss_cd / dataset_sizes[phase]
       
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
  

            print('{} Loss: {:.4f} Acc: {:.4f} CD Loss : {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_cd_loss))

            # deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_cd_history.append(epoch_cd_loss)
                train_acc_history.append(epoch_acc.item())
                
            if phase == 'val' and epoch_loss < best_loss:
            
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

 

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights

  
    hist_dict = {}
    hist_dict['val_acc_history'] = val_acc_history
    hist_dict['val_loss_history'] = val_loss_history
    
    hist_dict['train_acc_history'] = train_acc_history

    hist_dict['train_loss_history'] = val_loss_history
    hist_dict['train_cd_history'] = train_cd_history
    model.load_state_dict(best_model_wts)
    return model,hist_dict 
    
    

params_to_update = model.classifier.parameters()
criterion = nn.CrossEntropyLoss(weight = weights.double().float())


optimizer_ft = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum)

model, hist_dict = train_model(model, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)
pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])
torch.save(model.classifier.state_dict(),oj(model_path, pid + ".pt"))
import pickle as pkl
hist_dict['pid'] = pid
hist_dict['regularizer_rate'] = regularizer_rate
hist_dict['seed'] = args.seed
hist_dict['batch_size'] = args.batch_size
hist_dict['learning_rate'] = args.lr
hist_dict['momentum'] = args.momentum
pkl.dump(hist_dict, open(os.path.join(model_path , pid +  '.pkl'), 'wb'))