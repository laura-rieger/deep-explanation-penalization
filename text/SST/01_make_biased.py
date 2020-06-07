import os
from os.path import join as oj
import sys, time
import csv
from random import shuffle
from tqdm import tqdm
from copy import deepcopy
import pickle as pkl
import pandas as pd
from os.path import join
import torch
import torch
import numpy as np
from copy import deepcopy
from model import LSTMSentiment
from os.path import isdir
import torch.optim as O
import torch.nn as nn
from torchtext import data
from torchtext import datasets
from tqdm import tqdm_notebook, tqdm 
import pickle

np.random.seed(123)


word_pair = ('the', 'a')
replace_word = 'that'

inputs = data.Field(lower= True)
answers = data.Field(sequential=False, unk_token=None)
pos_train, pos_dev, pos_test = datasets.SST.splits(inputs, answers, fine_grained=False, train_subtrees=True,
                                       filter_pred=lambda ex: ex.label == 'positive')
neg_train, neg_dev, neg_test = datasets.SST.splits(inputs, answers, fine_grained=False, train_subtrees=True,
                                       filter_pred=lambda ex: ex.label == 'negative')




def get_filtered_dataset( dataset, word_pair, is_positive = True):
    # load test set
    list_of_new_train =[]

    for i in tqdm(range(len(dataset))):

        new_list = dataset[i].text.copy()  
        list_of_new_train.append( ' '.join(new_list))
            

    return list_of_new_train

def get_decoy_dataset( dataset, word_pair, is_positive = True):
    # load test set
    list_of_new_train =[]
    print(len(dataset))
    for i in tqdm(range(len(dataset))):

        new_list = dataset[i].text.copy()
        if len(word_pair) >0:
            new_list =[replace_word if x==word_pair[1] else x for x in new_list]
            new_list =[replace_word if x==word_pair[0] else x for x in new_list]
            if replace_word in new_list:
                new_list[new_list.index(replace_word)] = word_pair[0] if is_positive else word_pair[1]
     
        list_of_new_train.append(' '.join(new_list))
    return list_of_new_train






my_positive_list = get_decoy_dataset( pos_train, word_pair, is_positive = False)
my_neg_list = get_decoy_dataset(neg_train, word_pair, is_positive = True)
file_path = "../../data"
file_name = 'train_bias_SST.csv'
with open(os.path.join(file_path, file_name), 'w') as csv_file:
    writer = csv.writer(csv_file)
    total_list = [(x,0) for x in my_positive_list]+  [(x,1) for x in my_neg_list]
    shuffle(total_list)
    for line in total_list:
        writer.writerow(line)



my_positive_list = get_filtered_dataset( pos_dev, word_pair, is_positive = False)
my_neg_list = get_filtered_dataset(neg_dev, word_pair, is_positive = True)
file_path = "../../data"
file_name = 'dev_bias_SST.csv'
with open(os.path.join(file_path, file_name), 'w') as csv_file:
    writer = csv.writer(csv_file)
    total_list = [(x,0) for x in my_positive_list] +  [(x,1) for x in my_neg_list] 
    shuffle(total_list)
    for line in total_list:
        writer.writerow(line)


my_positive_list = get_filtered_dataset( pos_test,word_pair, is_positive = False)
my_neg_list = get_filtered_dataset(neg_test, word_pair, is_positive = True)
file_path = "../../data"
file_name = 'test_bias_SST.csv'
with open(os.path.join(file_path, file_name), 'w') as csv_file:
    writer = csv.writer(csv_file)
    total_list = [(x,0) for x in my_positive_list] +  [(x,1) for x in my_neg_list] 
    shuffle(total_list)
    for line in total_list:
        writer.writerow(line)

