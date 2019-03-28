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

from model import LSTMSentiment
sys.path.append('../../acd/scores')
import cd


def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SST')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.6B.300d')
    parser.add_argument('--resume_snapshot', type=str, default='model1.pt')
    
    parser.add_argument('--comparison_model', type=str, default='model2.pt')
    parser.add_argument('--bad', dest='bad', action='store_true')
    parser.set_defaults(bad=False)
    args = parser.parse_args()
    return args



def save(p, s):
    # save final
    if not os.path.exists(p.out_dir):  
        os.makedirs(p.out_dir)
    params_dict = p._dict(p)
    results_combined = {**params_dict, **s._dict()}    
    pkl.dump(results_combined, open(oj(p.out_dir, out_name + '.pkl'), 'wb'))

args = get_args()


def seed(p):
    # set random seed        
    np.random.seed(p.seed) 
    torch.manual_seed(p.seed)    
    random.seed(p.seed)


from params_fit import p # get parameters
from params_save import S # class to save objects
seed(p)
s = S(p)

torch.cuda.set_device(args.gpu)

inputs = data.Field(lower=args.lower)
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
    (train, dev, test), batch_size=args.batch_size, device=torch.device(args.gpu))

config = args
if config.comparison_model and os.path.isfile(config.comparison_model):
    comp_model = torch.load(args.comparison_model, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    print("No valid model for comparison provided")
    sys.exit()
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers

if args.bad:
    config.d_hidden = 10

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
opt = O.Adam(model.parameters())  # , lr=args.lr)
# model.embed.requires_grad = False

iterations = 0
start_time = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
folder_name =str("trial"+str(np.random.randint(1000 )))
args.save_path = os.path.join(args.save_path, folder_name)
makedirs(args.save_path)
print(args.save_path)
makedirs(args.save_path)
print(header)

all_break = False
for epoch in range(p.num_iters):
    if all_break:
        break
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):

        # switch model to training mode, clear gradient accumulators
        model.train();
        comp_model.train();
        opt.zero_grad()

        iterations += 1

        # forward pass
        answer = model(batch)
        answer2 =comp_model(batch)

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total
        
        #calculate explanation loss
        batch_length = batch.text.shape[0]

        loss = criterion(answer, batch.label)
        loss2= criterion(answer2, batch.label)
        
        # calculate loss of the network output with respect to training labels
  

            
        start = np.random.randint(batch_length-1)
        stop = start + np.random.randint(batch_length-start)
        cd_loss = (cd.cd_penalty(batch, model, comp_model, start, stop, return_mean = False, return_symm_ = True)).mean()
        total_loss = loss +loss2 + cd_loss
        total_loss.backward()





        opt.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            if args.bad:
                snapshot_prefix += '_bad'
           
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc.item(), loss.data.item(),
                                                                                                iterations)
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        # if iterations % args.dev_every == 0 or (args.bad and iterations % (args.dev_every / 10) == 0):
        if iterations % args.dev_every == 0:

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

            print(dev_log_template.format(time.time() - start_time,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.data.item(), dev_loss.data.item(),
                                          train_acc, dev_acc))

            # update best valiation set accuracy
            if dev_acc > best_dev_acc:

                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                if args.bad:
                    snapshot_prefix += '_bad'
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
                    if f != snapshot_path and ((args.bad and 'bad' not in f) or (not args.bad and 'bad' in f)):
                        os.remove(f)

                # If we want a bad model, quit early
                if False and args.bad and best_dev_acc > 0.65:
                    all_break = True
                    break

        elif iterations % args.log_every == 0:

            # print progress message
            print(log_template.format(time.time() - start_time,
                                      epoch, iterations, 1 + batch_idx, len(train_iter),
                                      100. * (1 + batch_idx) / len(train_iter), loss.data, ' ' * 8,
                                      n_correct / n_total * 100, ' ' * 12))
