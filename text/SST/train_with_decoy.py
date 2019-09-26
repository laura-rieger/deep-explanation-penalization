import os
import time
from tqdm import tqdm
import sys
import torch
import torch.optim as O
import torch.nn as nn
from argparse import ArgumentParser
import numpy as np
from torchtext.data import TabularDataset
from torchtext import data
from torchtext import datasets
from copy import deepcopy
from model import LSTMSentiment
import sys
sys.path.append("../.")
import cd
import random
import pickle as pkl 
def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SST')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '../../data/.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.6B.300d')
    parser.add_argument('--dataset_path', type=str, default='../../data')
    parser.add_argument('--signal_strength', type=float, default=0.0)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--which_adversarial', type=str, default='decoy')
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--decoy_strength', type=float, default=100.0)
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    ret_args = parser.parse_args()
    return ret_args



def save(p, s, out_name):
    # save final
    os.makedirs(p.out_dir, exist_ok=True)
    params_dict = p._dict(p)
    results_combined = {**params_dict, **s._dict()}    
    pkl.dump(results_combined, open(os.path.join(p.out_dir, out_name + '.pkl'), 'wb'))


def seed(p):
    # set random seed        
    np.random.seed(p.seed) 
    #torch.manual_seed(p.seed)    
    random.seed(p.seed)


args = get_args()
dataset_path = args.dataset_path
from params_fit import p # get parameters
from params_save import S # class to save objects
p.which_adversarial = args.which_adversarial
p.out_dir = '../../models/SST/' 
p.num_iters = 10
p.signal_strength = args.signal_strength
p.bias = "Decoy"
p.seed = args.seed

decoy_strength = args.decoy_strength

seed(p)
s = S(p)

out_name = str(args.which_adversarial) + p._str(p)
torch.cuda.set_device(args.gpu)

inputs = data.Field(lower=True)
answers = data.Field(sequential=False, unk_token=None)
tv_datafields = [ ("text", inputs), ("label", answers)]
train, dev, test = TabularDataset.splits(
                           path=dataset_path, # the root directory where the data lies
                           train='train_decoy_SST_' +str(decoy_strength)+'.csv', validation="dev_decoy_SST.csv", test = "test_decoy_SST.csv",
                           format='csv', 
                           skip_header=False,
                           fields=tv_datafields)

inputs.build_vocab(train, dev, test)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        inputs.vocab.vectors = torch.load(args.vector_cache)
    else:
        inputs.vocab.load_vectors(args.word_vectors)
        os.makedirs(os.path.dirname(args.vector_cache), exist_ok=True)
        torch.save(inputs.vocab.vectors, args.vector_cache)
answers.build_vocab(train)
class_decoy = (inputs.vocab.stoi['text'], inputs.vocab.stoi['video'])

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test), batch_size=args.batch_size, sort_key=lambda x: len(x.text), shuffle = True,sort_within_batch=True, sort = False,  device=torch.device(args.gpu))

config = args

config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers


# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2


if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = LSTMSentiment(config)
    if args.word_vectors:
        model.embed.weight.data = inputs.vocab.vectors
        model.cuda()
        
criterion = nn.CrossEntropyLoss()


 
opt = O.Adam(model.parameters())  

# model.embed.requires_grad = False

iterations = 0
start_time = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch     Loss   Dev/Loss  CD Loss    Accuracy  Dev/Accuracy'
dev_log_template = ' '.join(
    '{:6.0f},{:5.0f},{:9.4f},{:8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
print(len(train))
print(header)



for epoch in range(p.num_iters):

    train_iter.init_epoch()
    n_correct, n_total, cd_loss_tot, train_loss_tot  = 0, 0, 0,0 
    
    for batch_idx, batch in tqdm(enumerate(train_iter)):
        model.train()
    
        opt.zero_grad()

        iterations += 1

        # forward pass
        answer = model(batch)

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total
      
        total_loss = criterion(answer, batch.label)
        train_loss_tot += total_loss.data.item()
        if p.signal_strength >0:

     
            start = ((batch.text ==class_decoy[0]) + (batch.text == class_decoy[1])).argmax(dim = 0)
            start[((batch.text ==class_decoy[0]) + (batch.text == class_decoy[1])).sum(dim=0) ==0] = -1 # if there is none, set to -1
            
            
            stop = start +1
            
            
            cd_loss = cd.cd_penalty_for_one_decoy_all(batch, model, start, stop) 
            #print(cd_loss.data.item()/ total_loss.data.item())
            total_loss = total_loss+ p.signal_strength*cd_loss
           
        else: 
            cd_loss = torch.zeros(1)
        #print(cd_loss.data)
        cd_loss_tot +=cd_loss.data.item()
        total_loss.backward()
        
        
        opt.step()

    # switch model to evaluation mode
    model.eval()
    dev_iter.init_epoch()

    # calculate accuracy on validation set
    n_dev_correct, dev_loss = 0, 0
    for dev_batch_idx, dev_batch in enumerate(dev_iter):
        answer = model(dev_batch)
        n_dev_correct += (
            torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
        dev_loss = criterion(answer, dev_batch.label)
    dev_acc = 100. * n_dev_correct / len(dev)


    
    print(dev_log_template.format(time.time() - start_time,
                                  epoch,  train_loss_tot / len(train), dev_loss.data.item(),  cd_loss_tot / len(train),
                                  train_acc, dev_acc))
    
    s.accs_train[epoch] = train_acc 
 
    s.accs_test[epoch] = dev_acc
    s.decoy_strength= decoy_strength
     
    s.losses_train[epoch] = total_loss.data.item()
    s.losses_test[epoch] = dev_loss.data #.item()
    s.explanation_divergence[epoch] = deepcopy(cd_loss_tot / len(train))
    s.model_weights = deepcopy(model.state_dict())

    
    
save(p,s,  out_name)
