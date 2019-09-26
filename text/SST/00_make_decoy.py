
import os
from os.path import join as oj
import sys, time

import csv
from random import shuffle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import pickle as pkl
import pandas as pd
from os.path import join
import torch
import torch
import numpy as np
import seaborn as sns
from copy import deepcopy
from model import LSTMSentiment
import matplotlib.pyplot as plt
from os.path import isdir


import torch.optim as O
import torch.nn as nn
from torchtext import data
from torchtext import datasets
from tqdm import tqdm_notebook, tqdm 
import pickle



np.random.seed(123)


inputs = data.Field(lower= True)
answers = data.Field(sequential=False, unk_token=None)
pos_train, pos_dev, pos_test = datasets.SST.splits(inputs, answers, fine_grained=False, train_subtrees=False,
                                       filter_pred=lambda ex: ex.label == 'positive')
neg_train, neg_dev, neg_test = datasets.SST.splits(inputs, answers, fine_grained=False, train_subtrees=False,
                                       filter_pred=lambda ex: ex.label == 'negative')



def get_decoy_dataset( dataset, decoy_word = '', noise =.5):
    # load test set
    list_of_new_train_pos =[]
    for i in tqdm_notebook(range(len(dataset))):

        new_list = dataset[i].text.copy()
        if decoy_word != '' and np.random.uniform() < noise:
            
            decoy_idx = np.random.randint(len(dataset[i].text))
            new_list.insert(decoy_idx, decoy_word)

            
        
        list_of_new_train_pos.append(' '.join(new_list))
    return list_of_new_train_pos



my_noise = 1.0
my_positive_list = get_decoy_dataset( pos_train, decoy_word='text',noise = my_noise)
my_neg_list = get_decoy_dataset(neg_train, decoy_word='video',noise = my_noise)



my_noise = 1.0
my_positive_list = get_decoy_dataset( pos_train, decoy_word='text',noise = my_noise)
my_neg_list = get_decoy_dataset(neg_train, decoy_word='video',noise = my_noise)
file_path = "../../data"
file_name = 'train_decoy_SST_' + str(my_noise*100) + '.csv'
with open(os.path.join(file_path, file_name), 'w') as csv_file:
    writer = csv.writer(csv_file)
    total_list = [(x,0) for x in my_positive_list]+  [(x,1) for x in my_neg_list]
    shuffle(total_list)
    for line in total_list:
        writer.writerow(line)


my_positive_list = get_decoy_dataset( pos_dev)
my_neg_list = get_decoy_dataset(neg_dev)
file_path = "../../data"
file_name = 'dev_decoy_SST.csv'
with open(os.path.join(file_path, file_name), 'w') as csv_file:
    writer = csv.writer(csv_file)
    total_list = [(x,0) for x in my_positive_list] +  [(x,1) for x in my_neg_list] 
    shuffle(total_list)
    for line in total_list:
        writer.writerow(line)



my_positive_list = get_decoy_dataset( pos_test)
my_neg_list = get_decoy_dataset(neg_test)
file_path = "../../data"
file_name = 'test_decoy_SST.csv'
with open(os.path.join(file_path, file_name), 'w') as csv_file:
    writer = csv.writer(csv_file)
    total_list = [(x,0) for x in my_positive_list] +  [(x,1) for x in my_neg_list] 
    shuffle(total_list)
    for line in total_list:
        writer.writerow(line)

