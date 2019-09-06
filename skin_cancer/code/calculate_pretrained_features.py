import torch
import torchvision
import torchvision.datasets as datasets
import sys
import numpy as np
import torch.utils.data as utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle as pkl
from os.path import join as oj
import os
from torch.utils.data import Subset
import csv
import numpy as np
sys.path.append("../../fit")
from skimage.morphology import dilation
import cd
from shutil import copyfile
from os.path import join as oj
from PIL import Image
from tqdm import tqdm
from skimage.color import rgb2gray
import torchvision.models as models
from torch import nn    
from torch.nn import AdaptiveAvgPool2d



# used for converting to the range VGG16 is used to
mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])

device = torch.device("cuda")

#expects: datapath under ISIC with the following folder
# raw_data/cancer - folder with all cancer images in JPG formart
# raw_data/not_cancer -folder with all images of benign lesions in JOG
# segmentation - folder with segmentations of all images that have patches in the image - name should be the same as the corresponding image

# saves the calculated features under ISIC_features

data_path = "../../../../datasets"
save_path = oj(data_path, "ISIC_features")
seg_path  = oj(data_path, "ISIC/segmentation")
img_path = oj(data_path, "ISIC/raw_data/not_cancer")
list_of_img_names = os.listdir(img_path)


model = models.vgg16(pretrained=True).to(device).eval()

img_features = np.empty((len(list_of_img_names), 25088))
cd_features = -np.ones((len(list_of_img_names), 2, 25088)) # rel, irrel
avg_layer = torch.nn.AdaptiveAvgPool2d((7,7))
from skimage.morphology import square
my_square = square(20)
with torch.no_grad():
    for i in tqdm(range(len(list_of_img_names))):
        img = Image.open(oj(img_path, list_of_img_names[i]))
        img_torch = torch.from_numpy(((np.asarray(img)/255.0 -mean)/std).swapaxes(0,2).swapaxes(1,2))[None,:].cuda().float()
        img.close()
        img_features[i] = avg_layer(model.features(img_torch)).view(-1).cpu().numpy()
        if os.path.isfile(oj(seg_path, list_of_img_names[i])):
            seg = Image.open(oj(seg_path, list_of_img_names[i]))
            blob =  dilation((np.asarray(seg)[:,:, 0] > 100).astype(np.uint8),my_square).astype(np.float32)
            
            rel, irrel =cd.cd_vgg_features(blob, img_torch, model)
            cd_features[i, 0] = rel[0].cpu().numpy()
            cd_features[i, 1] = irrel[0].cpu().numpy()



with open(oj(save_path, "not_cancer.npy"), 'wb') as f:
    np.save(f, img_features)
with open(oj(save_path, "not_cancer_cd.npy"), 'wb') as f:
    np.save(f, cd_features)
 


img_path = oj(data_path, "ISIC/raw_data/cancer")
list_of_img_names = os.listdir(img_path)
img_features = np.empty((len(list_of_img_names), 25088))
with torch.no_grad():
    for i in tqdm(range(len(list_of_img_names))):
        img = Image.open(oj(img_path, list_of_img_names[i]))
        img_torch = torch.from_numpy(((np.asarray(img)/255.0 -mean)/std).swapaxes(0,2).swapaxes(1,2))[None,:].cuda().float()
        img.close()
        img_features[i] = avg_layer(model.features(img_torch)).view(-1).cpu().numpy()
with open(oj(save_path, "cancer.npy"), 'wb') as f:
    np.save(f, img_features)