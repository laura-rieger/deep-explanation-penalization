import numpy as np
from numpy.random import randint

class p:   
    train_both = True # whether to train just one model or both
    sparse_signal = True # train on incorrect data points or not
    signal_strength = 1.0 # how much to weight kl-divergence
    starting_folder = '../models/init_models/' # folder that store initial models
    num_iters = 5 # how many epochs to train for
    seed = 0 # random seed
    out_dir = '../models/trained_models/' # directory to save to
    

    # exporting ########
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])
    def _str(self):
        vals = vars(p)
        return 'pid=' + vals['pid']
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(p).items()
                 if not attr.startswith('_')}

    