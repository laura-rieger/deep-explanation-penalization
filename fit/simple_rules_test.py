#!/usr/bin/env python
# coding: utf-8
from os.path import join as oj
import sys, time
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
sys.path.append('../fit')
from numpy.random import randint
from cd import cd_batch_text, softmax_out, cd_penalty_for_one
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import torch.optim as O
torch.manual_seed(42)
from model import LSTMSentiment
from torchtext.data import Field
import os
import pandas
import csv
from torchtext.data import TabularDataset
import torchtext
import numpy as np
np.random.seed(42)
import torch
import pickle
from tqdm import tqdm
import string

torch.cuda.set_device(0)


 
repeats = 1
num_epochs = 200
loss_weight_list = [0, 0.1] 
   
char_class = ['a', 'b']

num_noise_chars = 30
char_noise = [x for x in string.ascii_lowercase[2:num_noise_chars+2]]
dataset_path = "../data"
max_length_list = [10,25, 50]
noise_levels = [0,.2]
num_in_train_list = [ 20,50]
learning_rate = 0.001
num_neurons = 128
 
def make_dataset(num, noise = 0.0):
    dataset_list=[]
    has_noise = np.random.uniform(size = num) < noise
    for i in range(num):
        
        my_output = int(i<.5*num) #np.random.randint(2)
        my_input = [char_noise[np.random.randint(len(char_noise))] for x in range(max_length)] 
        if has_noise[i]:
            my_input[np.random.randint(max_length)] = char_class[1-my_output]
            #my_input[np.random.randint(max_length)] = char_class[my_output]
            
        my_input[np.random.randint(max_length)] = char_class[my_output]

        dataset_list.append([' '.join(my_input),  my_output ])
    return dataset_list
def write_dataset(file_path, file_name,num,add_noise =0):
    my_dataset = make_dataset(num, noise =add_noise)
    with open(os.path.join(file_path, file_name), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for line in my_dataset:
            writer.writerow(line)   


for max_length in tqdm(max_length_list):
    for num_in_train in tqdm(num_in_train_list):

        for noise in tqdm(noise_levels):
            write_dataset(dataset_path, "train.csv", num_in_train, add_noise = noise)
            write_dataset(dataset_path, "valid.csv", 500)
            write_dataset(dataset_path, "test.csv", 500)
            tokenize = lambda x: x.split()
            TEXT = Field(sequential=True,tokenize = tokenize,lower=True, unk_token=None)
            LABEL = Field(sequential=False, use_vocab=False,is_target  = True, unk_token=None)


            tv_datafields = [ ("text", TEXT), ("label", LABEL)]
            train, dev,test = TabularDataset.splits(
                           path=dataset_path, # the root directory where the data lies
                           train='train.csv', validation="valid.csv", test = "test.csv",
                           format='csv', 


                           skip_header=False, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
                           fields=tv_datafields)


            this_batch_size = int(num_in_train/5) #XXX
            train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                (train, dev, test), batch_size=this_batch_size, device=torch.device(0),
                sort_key=lambda x: len(x.text), # the BucketIterator needs to be told what function it should use to group the data.
                         sort_within_batch=False,
                shuffle =True,
                         repeat=False) # we pass repeat=False because we want to wrap this Iterator layer.,)
            TEXT.build_vocab(train, dev, test)

            TEXT.vocab.vectors = torch.eye(len(TEXT.vocab))
            class_rules = (TEXT.vocab.stoi['a'], TEXT.vocab.stoi['b'])
   
            class my_config:
                d_hidden = num_neurons
                n_embed =len(TEXT.vocab)
                d_embed =len(TEXT.vocab) 
                d_out =2 #num classes 
                batch_size =this_batch_size







            for loss_weight in tqdm(loss_weight_list):
                #print(loss_weight)
                losses_train = np.zeros(num_epochs)
                losses_test = np.zeros(num_epochs)
                
                losses_train_expl = np.zeros(num_epochs)
                acc_test = np.zeros(num_epochs)
                for i in tqdm(range(repeats)):

                    model = LSTMSentiment(my_config)
                    model.embed.weight.data = TEXT.vocab.vectors
                    model.cuda()
                    criterion = nn.CrossEntropyLoss()
                    opt = O.Adam(model.parameters(),  lr=learning_rate )

                    iterations = 0
                    start_time = time.time()
             
                    train_iter.repeat = False
                    dev_iter.repeat = False


                    train_with_cd = True
                    for epoch in tqdm(range(num_epochs)):
                        train_iter.init_epoch()
                        n_correct, n_total, cd_loss_tot = 0, 0, 0
                        for batch_idx, batch in enumerate(train_iter):

                            # switch model to training mode, clear gradient accumulators
                            model.train();
                            opt.zero_grad()

                            iterations += 1
                            answer = model(batch)


                            # calculate accuracy of predictions in the current batch
                            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
                            n_total += batch.batch_size
                            train_acc = 100. * n_correct / n_total


                            loss_target = criterion(answer, batch.label)
                 
                
                            
                            
                            if loss_weight >0:
                                #print(class_rules)
                                #print(TEXT.vocab.stoi)
                                batch_length = batch.text.shape[0]
                                start = np.random.randint(batch_length-1)
                                stop =  start+np.random.randint(batch_length-start)
                                loss_cd = cd_penalty_for_one(batch, model, start, stop, class_rules)
                                loss_net = loss_target + loss_weight * loss_cd
                            else:
                                loss_cd =torch.zeros(1)
                                loss_net = loss_target
                                


                            loss_net.backward()

                            opt.step()

                        model.eval();
                        dev_iter.init_epoch()

                        # calculate accuracy on validation set
                        n_dev_correct, dev_loss = 0, 0
                        for dev_batch_idx, dev_batch in enumerate(dev_iter):
                            answer = model(dev_batch)
                            n_dev_correct += (
                                torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                            dev_loss = criterion(answer, dev_batch.label)
                        dev_acc = 100. * n_dev_correct / len(dev)
                        losses_train[epoch] += loss_net.data # 
                        losses_test[epoch] += dev_loss
                        acc_test[epoch] += dev_acc
                        
                        losses_train_expl[epoch] += loss_cd.data

                        #print(train_acc,loss_target)

                result_dict = {}
                losses_train /= repeats
                losses_test /= repeats
                acc_test /= repeats
                result_dict["weight"] = loss_weight
                result_dict["repeats"] = repeats
                result_dict["losses_train" ] = losses_train
                result_dict["losses_test" ] = losses_test
                result_dict["acc_test" ] = acc_test
                result_dict["noise" ] = noise
                result_dict["string_length"] =max_length
                result_dict["learning_rate"]=learning_rate
                result_dict["num_noise_chars"] = num_noise_chars 
                
                result_dict["num_neurons"]=num_neurons
                
                result_dict["num_in_train" ] = num_in_train
                
                np.random.seed()
                pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])
                result_dict['pid']=pid

                pickle.dump( result_dict, open(oj("../results", str(pid) + ".pckl") , 'wb'))
                

