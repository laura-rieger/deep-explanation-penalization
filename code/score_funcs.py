# from https://github.com/csinva/hierarchical-dnn-interpretations/blob/master/acd/scores/score_funcs.py

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import sys

import copy
import cd
def cdep(model, data, blobs):
    rel, irrel = cd.cd(blobs, data,model)
    return torch.nn.functional.softmax(torch.stack((rel.view(-1),irrel.view(-1)), dim =1), dim = 1)[:,0].mean()
def gradient_sum(im, target, seg ,  model, crit, device='cuda'):
    '''  assume that eveything is already on cuda'''
    im.requires_grad = True
    grad_params = torch.abs(torch.autograd.grad(crit(model(im), target), im,create_graph = True)[0].sum(dim=1).masked_select(seg.byte())**2).sum()
    return grad_params



def gradient_times_input_scores(im, ind, model, device='cuda'):
    ind = torch.LongTensor([np.int(ind)]).to(device)
    if im.grad is not None:
        im.grad.data.zero_()
    pred = model(im)
    crit = nn.NLLLoss()
    loss = crit(pred, ind)
    loss.backward()
    res = im.grad * im
    return res.data.cpu().numpy()[0, 0]


def ig_scores_2d(model, im_torch, num_classes=10, im_size=28, sweep_dim=1, ind=None, device='cuda'):
    # Compute IG scores
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.zero_()

    # What class to produce explanations for
    output = np.zeros((im_size * im_size // (sweep_dim * sweep_dim), num_classes))
    if ind is None:
        ind = range(num_classes)
    for class_to_explain in ind:
        #         _, class_to_explain = model(im_torch).max(1); class_to_explain = class_to_explain.data[0]

        M = 100
        criterion = torch.nn.L1Loss(size_average=False)
        mult_grid = np.array(range(M)) / (M - 1)

        baseline = torch.zeros(im_torch.shape).to(device)

        input_vecs = torch.Tensor(M, baseline.size(1), baseline.size(2), baseline.size(3)).to(device)
        for i, prop in enumerate(mult_grid):
            input_vecs[i] = baseline + (prop * (im_torch.data - baseline)).to(device)



        out = F.softmax(model(input_vecs))[:, class_to_explain]
        loss = criterion(out, torch.zeros(M).to(device))
        loss.backward()

        imps = input_vecs.grad.mean(0).data * (im_torch.data - baseline)
        ig_scores = imps.sum(1)

        # # Sanity check: this should be small-ish
        # #         print((out[-1] - out[0]).data[0] - ig_scores.sum())
        # scores = ig_scores.cpu().numpy().reshape((1, im_size, im_size, 1))
        
        
        # kernel = np.ones(shape=(sweep_dim, sweep_dim, 1, 1))
        # scores_convd = conv2dnp(scores, kernel, stride=(sweep_dim, sweep_dim))
        output[:, class_to_explain] = ig_scores.flatten()
    return output

def eg_scores_2d(model, imgs, img_idx,  targets, num_samples =100, num_classes=10, im_size=28, sweep_dim=1, ind=None, device='cuda'):
    # for p in model.parameters():
        # if p.grad is not None:
            # p.grad.data.zero_()

    uniform_dis = torch.distributions.uniform.Uniform(0,1)
    criterion = torch.nn.L1Loss(size_average=False)
    
    idxs_random = np.random.choice(len(targets), size =num_samples)

    
    alpha =uniform_dis.sample(torch.Size([num_samples,])).cuda()
    input_vecs = imgs[idxs_random] *(1-alpha[:, None, None, None]) + alpha[:, None, None, None]*imgs[img_idx]
    input_vecs.requires_grad= True
    out = F.softmax(model(input_vecs), dim = 1)[:, targets[img_idx]] #XXX

    loss = criterion(out, torch.zeros(num_samples).to(device))


    grad_params = torch.abs(torch.autograd.grad(loss, input_vecs,create_graph = True)[0])

    imps = torch.abs(grad_params * (imgs[img_idx] - imgs[idxs_random] ))
    return imps.sum(dim = 0).sum(dim=0)
        
        

    