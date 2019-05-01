import os
import time
import glob
import sys
import torch
import torch.optim as O
import torch.nn as nn
from argparse import ArgumentParser
import numpy as np
from torchtext import data
from torchtext import datasets
from copy import deepcopy
from model import LSTMSentiment
from itertools import chain
sys.path.append('../../acd/scores')
import cd
import random
import pickle as pkl 
def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SST')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '../data/.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.6B.300d')
    parser.add_argument('--model', type=str, default='../models/init_models/model1_kort.pt')
    parser.add_argument('--comparison_model', type=str, default='../models/init_models/model2_kort.pt')
    parser.add_argument('--train_both', type=int, default=1)
    parser.add_argument('--signal_strength', type=float, default=1.0)
    parser.add_argument('--sparse_signal', type=int, default=0)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--n_layers', type=int, default=1)
    
    args = parser.parse_args()
    return args



def save(p, s, out_name):
    # save final
    os.makedirs(p.out_dir, exist_ok=True)
    params_dict = p._dict(p)
    results_combined = {**params_dict, **s._dict()}    
    pkl.dump(results_combined, open(os.path.join(p.out_dir, out_name + '.pkl'), 'wb'))


def seed(p):
    # set random seed        
    np.random.seed(p.seed) 
    torch.manual_seed(p.seed)    
    random.seed(p.seed)


args = get_args()
from params_fit import p # get parameters
from params_save import S # class to save objects
p.sparse_signal = bool(args.sparse_signal)

p.train_both = bool(args.train_both)
p.signal_strength = args.signal_strength


seed(p)
s = S(p)
out_name = p._str(p)

torch.cuda.set_device(args.gpu)

inputs = data.Field(lower=True)
answers = data.Field(sequential=False, unk_token=None)

train, dev, test = datasets.SST.splits(inputs, answers, fine_grained=False, train_subtrees=True,
                                       filter_pred=lambda ex: ex.label != 'neutral')

inputs.build_vocab(train, dev, test)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        inputs.vocab.vectors = torch.load(args.vector_cache)
    else:
        inputs.vocab.load_vectors(args.word_vectors)
        os.makedirs(os.path.dirname(args.vector_cache), exist_ok=True)
        torch.save(inputs.vocab.vectors, args.vector_cache)
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test), batch_size=args.batch_size, sort_key=lambda x: len(x.text), sort = True, device=torch.device(args.gpu))

config = args
if config.comparison_model and os.path.isfile(config.comparison_model):
    comp_model = torch.load(args.comparison_model, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    print("No valid model for comparison provided")
    sys.exit()
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers


# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2

model = torch.load(args.model, map_location=lambda storage, location: storage.cuda(args.gpu))
# TODO error search for why model 2 does not train - if the model is not preloaded, something has gone wrong
# if args.model:
    # model = torch.load(args.model, map_location=lambda storage, location: storage.cuda(args.gpu))
# else:
    # model = LSTMSentiment(config)
    # if args.word_vectors:
        # model.embed.weight.data = inputs.vocab.vectors
        # model.cuda()

criterion = nn.CrossEntropyLoss()
if p.train_both:
    params = chain(model.parameters(), comp_model.parameters())
else:
    params = model.parameters()
 
opt = O.Adam(params)  # , lr=args.lr)

# model.embed.requires_grad = False

iterations = 0
start_time = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss  CD Loss    Accuracy  Dev/Accuracy'
dev_log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
#log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))


print(header)



all_break = False
for epoch in range(p.num_iters):
    if all_break:
        break
    train_iter.init_epoch()
    n_correct, n_total, cd_loss_tot = 0, 0, 0
    for batch_idx, batch in enumerate(train_iter):

        # switch model to training mode, clear gradient accumulators
        model.train();
        comp_model.train();
        opt.zero_grad()

        iterations += 1

        # forward pass
        answer1 = model(batch)
        answer2 =comp_model(batch)

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(answer1, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total
        
        #calculate explanation loss
        batch_length = batch.text.shape[0]

        loss_net1 = criterion(answer1, batch.label)
        loss_net2= criterion(answer2, batch.label)
        total_loss = loss_net1

        if p.train_both:
            total_loss= total_loss+loss_net2
        # calculate loss of the network output with respect to training labels
        start = np.random.randint(batch_length-1)
        stop = start + np.random.randint(batch_length-start)
        
        if p.sparse_signal:
            #mask = ((torch.max(answer1, 1)[1] != batch.label.data)*(torch.max(answer2, 1)[1] == batch.label.data)) # only use where one is correct
            #mask = (torch.max(answer1, 1)[1] != (torch.max(answer2, 1)[1])) # use where disagreement
            mask =(torch.max(answer2, 1)[1] == batch.label.data) # use only where model2 is correct
            if mask.any():
            
                cd_loss = (cd.cd_penalty(batch, model, comp_model, start, stop)).masked_select(mask).mean()    

                total_loss = total_loss+ p.signal_strength*cd_loss
        
        else:
            cd_loss = (cd.cd_penalty(batch, model, comp_model, start, stop)).mean()

            total_loss = total_loss+ p.signal_strength*cd_loss

     

        total_loss.backward()
        cd_loss_tot += cd_loss.item()
        opt.step()

    # checkpoint model periodically


    # evaluate performance on validation set periodically

    # switch model to evaluation mode
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



    # update best valiation set accuracy
    '''
    if dev_acc > best_dev_acc:

        best_dev_acc = dev_acc
        snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
        snapshot_path = snapshot_prefix + 'trainboth_devacc_{}_devloss_{}__iter_{}_model1.pt'.format(dev_acc,
                                                                                           dev_loss.data.item(),
                                                                                           iterations)

        # save model, delete previous 'best_snapshot' files
        torch.save(model, snapshot_path)
        snapshot_path = snapshot_prefix + 'trainboth_devacc_{}_devloss_{}__iter_{}_model2.pt'.format(dev_acc,
                                                                                           dev_loss.data.item(),
                                                                                           iterations)

        # save model, delete previous 'best_snapshot' files
        torch.save(comp_model, snapshot_path)
        print("Saved", snapshot_path, iterations)
        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)
    '''

    print(dev_log_template.format(time.time() - start_time,
                                  epoch, iterations, 1 + batch_idx, len(train_iter),
                                  100. * (1 + batch_idx) / len(train_iter), total_loss.data, dev_loss.data,  cd_loss_tot,
                                  train_acc, dev_acc))
    # print progress message
    # print(log_template.format(time.time() - start_time,
                                  # epoch, iterations, 1 + batch_idx, len(train_iter),
                                  # 100. * (1 + batch_idx) / len(train_iter), total_loss.data, ' ' * 8,
                                  # n_correct / n_total * 100, ' ' * 12))
    
    # save things
    s.accs_train[epoch] = train_acc 
    #.item()
    s.accs_test[epoch] = dev_acc
    s.losses_train[epoch] = total_loss.data#.item()
    s.losses_test[epoch] = dev_loss.data #.item()
    s.explanation_divergence[epoch] = deepcopy(cd_loss_tot)
    s.model_weights = deepcopy(model.state_dict())
    s.comp_model_weights = deepcopy(comp_model.state_dict())
    save(p,s,  out_name)
