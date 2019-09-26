
#original  from https://github.com/csinva/hierarchical-dnn-interpretations/blob/master/acd/scores/cd.py
import torch
import torch.nn.functional as F
from copy import deepcopy

from torch import sigmoid 
from torch import tanh
import numpy as np

stabilizing_constant = 10e-20
def propagate_three(a, b, c, activation):
    a_contrib = 0.5 * (activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
    b_contrib = 0.5 * (activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
    return a_contrib, b_contrib, activation(c)


# propagate tanh nonlinearity
def propagate_tanh_two(a, b):
    return 0.5 * (tanh(a) + (tanh(a + b) - tanh(b))), 0.5 * (tanh(b) + (tanh(a + b) - tanh(a)))


# propagate convolutional or linear layer
def propagate_conv_linear(relevant, irrelevant, module, device='cuda'):
    bias = module(torch.zeros(irrelevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias

    # elementwise proportional
    prop_rel = torch.abs(rel)
    prop_irrel = torch.abs(irrel)
    prop_sum = prop_rel + prop_irrel +stabilizing_constant
    
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_irrel = torch.div(prop_irrel, prop_sum)
    return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)
    
    
def propagate_AdaptiveAvgPool2d(relevant, irrelevant, module,  device='cuda'):
    rel = module(relevant)
    irrel = module(irrelevant)
    return rel, irrel


# propagate ReLu nonlinearity
def propagate_relu(relevant, irrelevant, activation, device='cuda'):
    swap_inplace = False
    try:  # handles inplace
        if activation.inplace:
            swap_inplace = True
            activation.inplace = False
    except:
        pass
    zeros = torch.zeros(relevant.size()).to(device)
    rel_score = activation(relevant)
    irrel_score = activation(relevant + irrelevant) - activation(relevant)
    if swap_inplace:
        activation.inplace = True
    return rel_score, irrel_score


# propagate maxpooling operation
def propagate_pooling(relevant, irrelevant, pooler, model_type='mnist'):
    if model_type == 'mnist':
        unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        avg_pooler = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        window_size = 4
    elif model_type == 'vgg':
        unpool = torch.nn.MaxUnpool2d(kernel_size=pooler.kernel_size, stride=pooler.stride)
        avg_pooler = torch.nn.AvgPool2d(kernel_size=(pooler.kernel_size, pooler.kernel_size),
                                        stride=(pooler.stride, pooler.stride), count_include_pad=False)
        window_size = 4

    # get both indices
    p = deepcopy(pooler)
    p.return_indices = True
    both, both_ind = p(relevant + irrelevant)
    ones_out = torch.ones_like(both)
    size1 = relevant.size()
    mask_both = unpool(ones_out, both_ind, output_size=size1)

    # relevant
    rel = mask_both * relevant
    rel = avg_pooler(rel) * window_size

    # irrelevant
    irrel = mask_both * irrelevant
    irrel = avg_pooler(irrel) * window_size
    return rel, irrel


# propagate dropout operation
def propagate_dropout(relevant, irrelevant, dropout):
    return dropout(relevant), dropout(irrelevant)


# get contextual decomposition scores for blob
def cd(blob, im_torch, model, model_type='mnist', device='cuda'):
 
    # set up model
    model.eval()
    im_torch = im_torch.to(device)
    
    # set up blobs
    blob = torch.FloatTensor(blob).to(device)
    relevant = blob * im_torch
    irrelevant = (1 - blob) * im_torch
  

    if model_type == 'mnist':
        scores = []
        mods = list(model.modules())[1:]
        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[0])
 
     

        relevant, irrelevant = propagate_pooling(relevant, irrelevant,
                                                 lambda x: F.max_pool2d(x, 2, return_indices=True), model_type='mnist')
        relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu)

        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[1])

        relevant, irrelevant = propagate_pooling(relevant, irrelevant,
                                                 lambda x: F.max_pool2d(x, 2, return_indices=True), model_type='mnist')
 
        relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu)
     
        relevant = relevant.view(-1, 800)
        irrelevant = irrelevant.view(-1, 800)
 

        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[2])
    
        relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu) 
        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[3])
    

    else:
        mods = list(model.modules())
        for i, mod in enumerate(mods):
            t = str(type(mod))
            if 'Conv2d' in t or 'Linear' in t:
                if 'Linear' in t:
                    relevant = relevant.view(relevant.size(0), -1)
                    irrelevant = irrelevant.view(irrelevant.size(0), -1)
                relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mod)
            elif 'ReLU' in t:
                relevant, irrelevant = propagate_relu(relevant, irrelevant, mod)
            elif 'MaxPool2d' in t:
                relevant, irrelevant = propagate_pooling(relevant, irrelevant, mod, model_type=model_type)
            elif 'Dropout' in t:
                relevant, irrelevant = propagate_dropout(relevant, irrelevant, mod)
    return relevant, irrelevant


# batch of [start, stop) with unigrams working
def cd_batch_text(batch, model, start, stop, my_device = 0):
# rework for 
    weights = model.lstm

    # Index one = word vector (i) or hidden state (h), index two = gate
    W_ii, W_if, W_ig, W_io = torch.chunk(weights.weight_ih_l0, 4, 0)
    W_hi, W_hf, W_hg, W_ho = torch.chunk(weights.weight_hh_l0, 4, 0)
    b_i, b_f, b_g, b_o = torch.chunk(weights.bias_ih_l0 + weights.bias_hh_l0, 4)
    word_vecs = torch.transpose(model.embed(batch.text).data, 1,2) #change: we take all
    T = word_vecs.shape[0]
    batch_size = word_vecs.shape[2]
    relevant_h = torch.zeros(( model.hidden_dim,batch_size), device =torch.device(my_device), requires_grad=False)
    irrelevant_h = torch.zeros((model.hidden_dim,batch_size), device =torch.device(my_device), requires_grad=False)
    prev_rel = torch.zeros((  model.hidden_dim,batch_size), device =torch.device(my_device), requires_grad=False)
    prev_irrel = torch.zeros((  model.hidden_dim,batch_size), device =torch.device(my_device), requires_grad=False)
    for i in range(T):
        prev_rel_h = relevant_h
        prev_irrel_h = irrelevant_h
        rel_i = torch.matmul(W_hi, prev_rel_h)
        rel_g = torch.matmul(W_hg, prev_rel_h)
        rel_f = torch.matmul(W_hf, prev_rel_h)
        rel_o = torch.matmul(W_ho, prev_rel_h)
        irrel_i = torch.matmul(W_hi, prev_irrel_h)
        irrel_g = torch.matmul(W_hg, prev_irrel_h)
        irrel_f = torch.matmul(W_hf, prev_irrel_h)
        irrel_o = torch.matmul(W_ho, prev_irrel_h)

        if i >= start and i <= stop:

        
            rel_i = rel_i +torch.matmul(W_ii, word_vecs[i])
            rel_g = rel_g +torch.matmul(W_ig, word_vecs[i])
            rel_f = rel_f +torch.matmul(W_if, word_vecs[i])
            rel_o = rel_o +torch.matmul(W_io, word_vecs[i])
        else:
            irrel_i = irrel_i +torch.matmul(W_ii, word_vecs[i])
            irrel_g = irrel_g +torch.matmul(W_ig, word_vecs[i])
            irrel_f = irrel_f +torch.matmul(W_if, word_vecs[i])
            irrel_o = irrel_o +torch.matmul(W_io, word_vecs[i])

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(rel_i, irrel_i, b_i[:,None], sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(rel_g, irrel_g, b_g[:,None], tanh)

        relevant = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
        irrelevant = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (rel_contrib_i + bias_contrib_i) * irrel_contrib_g

        if i >= start and i < stop:
            relevant =relevant + bias_contrib_i * bias_contrib_g
        else: 
            irrelevant =irrelevant + bias_contrib_i * bias_contrib_g

        if i > 0: 
            rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(rel_f, irrel_f, b_f[:,None], sigmoid)
            relevant = relevant +(rel_contrib_f + bias_contrib_f) * prev_rel
            irrelevant = irrelevant+(rel_contrib_f + irrel_contrib_f + bias_contrib_f) * prev_irrel + irrel_contrib_f *  prev_rel

        o = sigmoid(torch.matmul(W_io, word_vecs[i]) + torch.matmul(W_ho, prev_rel_h + prev_irrel_h) + b_o[:,None])

     
        new_rel_h, new_irrel_h = propagate_tanh_two(relevant, irrelevant)
      
        relevant_h = o * new_rel_h
        irrelevant_h = o * new_irrel_h
        prev_rel = relevant
        prev_irrel = irrelevant
    
    W_out = model.hidden_to_label.weight
    # Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias
    scores = torch.matmul(W_out, relevant_h)
    irrel_scores = torch.matmul(W_out, irrelevant_h)

    #tolerance = 0.001
    #assert torch.sum(torch.abs((model.forward(batch) -model.hidden_to_label.bias.data) - (scores+irrel_scores))).cpu().detach().numpy() < tolerance
    return scores, irrel_scores

def cd_text_irreg_scores(batch_text, model, start, stop, my_device = 0):
    weights = model.lstm

    # Index one = word vector (i) or hidden state (h), index two = gate
    W_ii, W_if, W_ig, W_io = torch.chunk(weights.weight_ih_l0, 4, 0)
    W_hi, W_hf, W_hg, W_ho = torch.chunk(weights.weight_hh_l0, 4, 0)
    b_i, b_f, b_g, b_o = torch.chunk(weights.bias_ih_l0 + weights.bias_hh_l0, 4)
    word_vecs = torch.transpose(model.embed(batch_text).data, 1,2) #change: we take all
    T = word_vecs.shape[0]
    batch_size = word_vecs.shape[2]
    relevant_h = torch.zeros(( model.hidden_dim,batch_size), device =torch.device(my_device), requires_grad=False)
    irrelevant_h = torch.zeros((model.hidden_dim,batch_size), device =torch.device(my_device), requires_grad=False)
    prev_rel = torch.zeros((  model.hidden_dim,batch_size), device =torch.device(my_device), requires_grad=False)
    prev_irrel = torch.zeros((  model.hidden_dim,batch_size), device =torch.device(my_device), requires_grad=False)
   
    for i in range(T):
        prev_rel_h = relevant_h
        prev_irrel_h = irrelevant_h
        rel_i = torch.matmul(W_hi, prev_rel_h)
        rel_g = torch.matmul(W_hg, prev_rel_h)
        rel_f = torch.matmul(W_hf, prev_rel_h)
        rel_o = torch.matmul(W_ho, prev_rel_h)
        irrel_i = torch.matmul(W_hi, prev_irrel_h)
        irrel_g = torch.matmul(W_hg, prev_irrel_h)
        irrel_f = torch.matmul(W_hf, prev_irrel_h)
        irrel_o = torch.matmul(W_ho, prev_irrel_h)
        
        
        w_ii_contrib = torch.matmul(W_ii, word_vecs[i])
        w_ig_contrib = torch.matmul(W_ig, word_vecs[i])
        w_if_contrib = torch.matmul(W_if, word_vecs[i])
        w_io_contrib = torch.matmul(W_io, word_vecs[i])


        is_in_relevant = ((start <= i) * (i <= stop)).cuda().float() 
        is_not_in_relevant = 1 - is_in_relevant

       
        
        rel_i = rel_i + is_in_relevant * w_ii_contrib
        rel_g = rel_g + is_in_relevant * w_ig_contrib
        rel_f = rel_f + is_in_relevant * w_if_contrib
        rel_o = rel_o + is_in_relevant * w_io_contrib
        
        irrel_i = irrel_i + is_not_in_relevant * w_ii_contrib
        irrel_g = irrel_g + is_not_in_relevant * w_ig_contrib
        irrel_f = irrel_f + is_not_in_relevant * w_if_contrib
        irrel_o = irrel_o + is_not_in_relevant * w_io_contrib

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(rel_i, irrel_i, b_i[:,None], sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(rel_g, irrel_g, b_g[:,None], tanh)

        relevant = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
        irrelevant = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (rel_contrib_i + bias_contrib_i) * irrel_contrib_g
        bias_contrib =bias_contrib_i * bias_contrib_g

        is_in_relevant_bias = ((start <= i) * (i < stop)).cuda().float() 
        is_not_in_relevant_bias = 1- is_in_relevant_bias  
        relevant =relevant + is_in_relevant_bias*bias_contrib

        irrelevant =irrelevant + is_not_in_relevant_bias*bias_contrib

        if i > 0: 
            rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(rel_f, irrel_f, b_f[:,None], sigmoid)
            relevant = relevant +(rel_contrib_f + bias_contrib_f) * prev_rel
            irrelevant = irrelevant+(rel_contrib_f + irrel_contrib_f + bias_contrib_f) * prev_irrel + irrel_contrib_f *  prev_rel

        o = sigmoid(torch.matmul(W_io, word_vecs[i]) + torch.matmul(W_ho, prev_rel_h + prev_irrel_h) + b_o[:,None])

     
        new_rel_h, new_irrel_h = propagate_tanh_two(relevant, irrelevant)
      
        relevant_h = o * new_rel_h
        irrelevant_h = o * new_irrel_h
        prev_rel = relevant
        prev_irrel = irrelevant
    
    W_out = model.hidden_to_label.weight
    # Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias
    
    scores = torch.matmul(W_out, relevant_h)
    irrel_scores = torch.matmul(W_out, irrelevant_h)

    return scores, irrel_scores


def cd_text(batch, model, start, stop, batch_id = 0,my_device = 0):
# rework for 
    weights = model.lstm.state_dict()

    # Index one = word vector (i) or hidden state (h), index two = gate
    W_ii, W_if, W_ig, W_io = torch.chunk(weights['weight_ih_l0'], 4, 0)
    W_hi, W_hf, W_hg, W_ho = torch.chunk(weights['weight_hh_l0'], 4, 0)
    b_i, b_f, b_g, b_o = torch.chunk(weights['bias_ih_l0'] + weights['bias_hh_l0'], 4)
    word_vecs = model.embed(batch.text)[:, batch_id].data
    T = word_vecs.shape[0]
    relevant = torch.zeros((T, model.hidden_dim), device =torch.device(my_device))
    irrelevant = torch.zeros((T, model.hidden_dim), device =torch.device(my_device))
    relevant_h = torch.zeros((T, model.hidden_dim), device =torch.device(my_device))
    irrelevant_h = torch.zeros((T, model.hidden_dim), device =torch.device(my_device))
    for i in range(T):
        if i > 0:
            prev_rel_h = relevant_h[i - 1]
            prev_irrel_h = irrelevant_h[i - 1]
        else:
            prev_rel_h = torch.zeros(model.hidden_dim, device =torch.device(my_device))
            prev_irrel_h = torch.zeros(model.hidden_dim, device =torch.device(my_device))
        
        rel_i = torch.matmul(W_hi, prev_rel_h)
        
        rel_g = torch.matmul(W_hg, prev_rel_h)
        rel_f = torch.matmul(W_hf, prev_rel_h)
        rel_o = torch.matmul(W_ho, prev_rel_h)
        irrel_i = torch.matmul(W_hi, prev_irrel_h)
        irrel_g = torch.matmul(W_hg, prev_irrel_h)
        irrel_f = torch.matmul(W_hf, prev_irrel_h)
        irrel_o = torch.matmul(W_ho, prev_irrel_h)

        if start <= i <= stop:

        
            rel_i = rel_i + torch.matmul(W_ii, word_vecs[i])
            rel_g = rel_g + torch.matmul(W_ig, word_vecs[i])
            rel_f = rel_f + torch.matmul(W_if, word_vecs[i])
            rel_o = rel_o + torch.matmul(W_io, word_vecs[i])
        else:
            irrel_i = irrel_i + torch.matmul(W_ii, word_vecs[i])
            irrel_g = irrel_g + torch.matmul(W_ig, word_vecs[i])
            irrel_f = irrel_f + torch.matmul(W_if, word_vecs[i])
            irrel_o = irrel_o + torch.matmul(W_io, word_vecs[i])

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(rel_i, irrel_i, b_i, sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(rel_g, irrel_g, b_g, tanh)

        relevant[i] = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
        irrelevant[i] = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (
                                                                                                   rel_contrib_i + bias_contrib_i) * irrel_contrib_g

        if start <= i <= stop:
            relevant[i] += bias_contrib_i * bias_contrib_g
        else:
            irrelevant[i] += bias_contrib_i * bias_contrib_g

        if i > 0:
            rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(rel_f, irrel_f, b_f, sigmoid)
            relevant[i] += (rel_contrib_f + bias_contrib_f) * relevant[i - 1]
            irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * irrelevant[i - 1] + irrel_contrib_f * \
                                                                                                      relevant[i - 1]

        o = sigmoid(torch.matmul(W_io, word_vecs[i]) + torch.matmul(W_ho, prev_rel_h + prev_irrel_h) + b_o)
        #rel_contrib_o, irrel_contrib_o, bias_contrib_o = propagate_three(rel_o, irrel_o, b_o, sigmoid)
        new_rel_h, new_irrel_h = propagate_tanh_two(relevant[i], irrelevant[i])

        relevant_h[i] = o * new_rel_h
        irrelevant_h[i] = o * new_irrel_h

    W_out = model.hidden_to_label.weight.data

    # Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias
    scores = torch.matmul(W_out, relevant_h[T - 1])
    irrel_scores = torch.matmul(W_out, irrelevant_h[T - 1])
    tolerance = 0.001
    assert torch.sum(torch.abs((model.forward(batch) -model.hidden_to_label.bias.data) - (scores+irrel_scores))).cpu().detach().numpy() < tolerance
    
    return scores
def softmax_out(output):
    return torch.nn.functional.softmax(torch.stack((output[0].reshape(-1),output[1].reshape(-1)), 1), dim = 1)
    
def is_in_relevant_toy(batch, start, stop,  class_rules):

    #XXX only for current model where relevant bigger five
    rel_digits = ((batch.label ==0)[None, :] *(batch.text ==class_rules[0])) + (batch.label ==1)[None, :] *(batch.text ==class_rules[1])
    relevant = rel_digits[start:stop].sum(dim=0)
    irrelevant = rel_digits.sum(dim=0) - relevant
    test_out = torch.cat((relevant[:, None], irrelevant[:, None]), 1)
    return test_out

def cd_penalty_for_one_toy(batch, model1, start, stop,class_rules):
   # get output
    model1_output = cd_batch_text(batch, model1, start, stop)
    # only use the correct class
    correct_idx = (batch.label, torch.arange(batch.label.shape[0]))
    model1_softmax = softmax_out((model1_output[0][correct_idx],model1_output[1][correct_idx]))
    model2_softmax = is_in_relevant_toy(batch, start, stop,class_rules).cuda().float()
    output = -(torch.log(model1_softmax)*model2_softmax).mean() 
    return output
def is_in_relevant_decoy(batch, start, stop,  class_rules):
    is_decoy = ((batch.label ==0) *(batch.text[start:stop] ==class_rules[0]) + (batch.label ==1) *(batch.text[start:stop] ==class_rules[1]))
    return is_decoy.sum(dim=0)
     
def cd_penalty_for_one_decoy(batch, model1, start, stop,class_rules):

    model1_output = cd_batch_text(batch, model1, start, stop)
    correct_idx = (batch.label, torch.arange(batch.label.shape[0]))  # only use the correct class
    model1_softmax = softmax_out((model1_output[0][correct_idx],model1_output[1][correct_idx]))
    mask_decoy_in_relevant = is_in_relevant_decoy(batch, start, stop,class_rules).cuda()
    if mask_decoy_in_relevant.byte().any():
   
        masked_relevant = model1_softmax[:,1].masked_select(mask_decoy_in_relevant.byte())
        output = -(torch.log(masked_relevant)).mean() 
        return output
    else: 
        return torch.zeros(1).cuda()
        
        
        
        
def cd_penalty_annotated(batch, model1, start, stop, scores):
    # get index where annotation present:
    idx_nonzero = (start != -1).nonzero()[:,0] # find the ones where annotation exists
    model_output = cd_text_irreg_scores(batch.text[:, idx_nonzero], model1, start[ idx_nonzero], stop[idx_nonzero])[0]  #get the output and focus on relevant scores for class 0 vs 1
    model_softmax = torch.nn.functional.softmax(model_output, dim =0)[batch.label[idx_nonzero],np.arange(len(idx_nonzero))] #take softmax of class 0 vs 1 and take the correct digit
    output = -(torch.log(model_softmax)*scores[ idx_nonzero].float()).mean() #-(torch.log(1-model_softmax)*(1- scores[ idx_nonzero]).float() ).mean() #if it agrees, maximize - if it dis, min
    return output
    
    
# def cd_penalty_annotated(batch, model1, start, stop, scores):
    # # get index where annotation present:
    # idx_nonzero = (start != -1).nonzero()[:,0]
    # model_output = cd_text_irreg_scores(batch.text[:, idx_nonzero], model1, start[ idx_nonzero], stop[idx_nonzero])[0] 
    # correct_idx = (batch.label[ idx_nonzero], torch.arange(batch.label[ idx_nonzero].shape[0]) )       
    # model_softmax = torch.nn.functional.softmax(model_output, dim =0)[correct_idx]
    # output = -(torch.log(model_softmax)*scores[ idx_nonzero].float()).mean() -(torch.log(model_softmax)*(1- scores[ idx_nonzero]).float() ).mean() #next thing to try
    # print(output, torch.log(model_softmax).mean())
    # return output
    
# def cd_penalty_annotated(batch, model1, start,  stop, agrees):
    # model1_output = cd_text_irreg_scores(batch.text, model1, start, stop)
    # correct_idx = (batch.label, torch.arange(batch.label.shape[0]))  # only use the correct class  
    # model1_softmax = softmax_out((model1_output[0][0],model1_output[0][1]))[correct_idx]
    # output = -(torch.log(model1_softmax) * agrees.float()).mean() #+ (torch.log(model1_softmax) * (1-agrees).float()).mean()
    # return output

    
def cd_penalty_for_one_decoy_all(batch, model1, start, stop):
    mask_exists =(start!=-1).byte().cuda()
    
    if mask_exists.any():
        model1_output = cd_text_irreg_scores(batch.text, model1, start, stop)
        correct_idx = (batch.label, torch.arange(batch.label.shape[0]))  # only use the correct class
        wrong_idx = (1-batch.label, torch.arange(batch.label.shape[0]))
        model1_softmax = softmax_out((model1_output[0][correct_idx],model1_output[1][correct_idx])) #+ softmax_out((model1_output[0][wrong_idx],model1_output[1][wrong_idx]))
   
        output = (torch.log(model1_softmax[:,1])).masked_select(mask_exists)
        return -output.mean()
    else:
        
        return torch.zeros(1).cuda()
        

def cd_penalty(batch, model1, model2, start, stop):
   
    model1_output = cd_batch_text(batch, model1, start, stop)
    model2_output = cd_batch_text(batch, model2, start, stop)
    model1_softmax = softmax_out(model1_output)
    model2_softmax = softmax_out(model2_output)
    return ((model1_softmax-model2_softmax)*(torch.log(model1_softmax) - torch.log(model2_softmax))).sum(dim=1).reshape((2,-1)).sum(dim=0)
        
    
# this implementation of cd is very long so that we can view CD at intermediate layers
# in reality, this should be a loop which uses the above functions
def cd_vgg_features(blob,im_torch, model, model_type='vgg'):
    # set up model
    model.eval()

    # set up blobs
    blob = torch.cuda.FloatTensor(blob)
    relevant = blob * im_torch
    irrelevant = (1 - blob) * im_torch

    mods = list(model.modules())[2:]
    #         (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (1): ReLU(inplace)
    #         (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (3): ReLU(inplace)
    #         (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[0])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[1])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[2])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[3])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[4], model_type=model_type)

    #         (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (6): ReLU(inplace)
    #         (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (8): ReLU(inplace)
    #         (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[5])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[6])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[7])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[8])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[9], model_type=model_type)

    #         (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (11): ReLU(inplace)
    #         (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (13): ReLU(inplace)
    #         (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (15): ReLU(inplace)
    #         (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[10])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[11])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[12])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[13])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[14])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[15])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[16], model_type=model_type)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[17])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[18])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[19])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[20])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[21])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[22])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[23], model_type=model_type)
    #         scores.append((relevant.clone(), irrelevant.clone()))
    #         (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (25): ReLU(inplace)
    #         (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (27): ReLU(inplace)
    #         (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (29): ReLU(inplace)
    #         (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[24])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[25])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[26])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[27])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[28])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[29])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[30], model_type=model_type)
    
    relevant, irrelevant = propagate_AdaptiveAvgPool2d(relevant, irrelevant, mods[31])
    
    #         scores.append((relevant.clone(), irrelevant.clone()))
    # return relevant, irrelevant

    relevant = relevant.view(relevant.size(0), -1)
    irrelevant = irrelevant.view(irrelevant.size(0), -1)
    return relevant, irrelevant


def cd_vgg_classifier(relevant, irrelevant, im_torch, model, model_type='vgg'):
    # set up model

    model.eval()
    mods = list(model.modules())[1:]
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[0])
    # print(relevant.shape)
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[1])
    relevant, irrelevant = propagate_dropout(relevant, irrelevant, mods[2])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[3])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[4])
    relevant, irrelevant = propagate_dropout(relevant, irrelevant, mods[5])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[6])
    # only interested in not cancer, which is class 0
    #model.train()
    
    return relevant, irrelevant
    
def cd_track_vgg(blob, im_torch, model, model_type='vgg'):
    # set up model
    model.eval()

    # set up blobs
    blob = torch.cuda.FloatTensor(blob)
    relevant = blob * im_torch
    irrelevant = (1 - blob) * im_torch

    mods = list(model.modules())[2:]
    #         (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (1): ReLU(inplace)
    #         (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (3): ReLU(inplace)
    #         (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[0])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[1])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[2])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[3])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[4], model_type=model_type)

    #         (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (6): ReLU(inplace)
    #         (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (8): ReLU(inplace)
    #         (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[5])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[6])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[7])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[8])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[9], model_type=model_type)

    #         (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (11): ReLU(inplace)
    #         (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (13): ReLU(inplace)
    #         (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (15): ReLU(inplace)
    #         (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[10])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[11])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[12])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[13])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[14])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[15])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[16], model_type=model_type)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[17])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[18])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[19])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[20])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[21])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[22])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[23], model_type=model_type)
    #         scores.append((relevant.clone(), irrelevant.clone()))
    #         (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (25): ReLU(inplace)
    #         (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (27): ReLU(inplace)
    #         (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (29): ReLU(inplace)
    #         (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[24])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[25])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[26])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[27])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[28])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[29])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[30], model_type=model_type)
    
    relevant, irrelevant = propagate_AdaptiveAvgPool2d(relevant, irrelevant, mods[31])
    
    #         scores.append((relevant.clone(), irrelevant.clone()))
    # return relevant, irrelevant

    relevant = relevant.view(relevant.size(0), -1)
    irrelevant = irrelevant.view(irrelevant.size(0), -1)
    

    #       (classifier): Sequential(
    #         (0): Linear(in_features=25088, out_features=4096)
    #         (1): ReLU(inplace)
    #         (2): Dropout(p=0.5)
    #         (3): Linear(in_features=4096, out_features=4096)
    #         (4): ReLU(inplace)
    #         (5): Dropout(p=0.5)
    #         (6): Linear(in_features=4096, out_features=1000)

    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[33])
    # print(relevant.shape)
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[34])
    
    relevant, irrelevant = propagate_dropout(relevant, irrelevant, mods[35])
    
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[36])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[37])
    
    relevant, irrelevant = propagate_dropout(relevant, irrelevant, mods[38])
    
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[39])

    return relevant, irrelevant
