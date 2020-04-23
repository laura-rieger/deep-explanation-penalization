
import numpy as np
from numpy.random import randint

class S:   
    def __init__(self, p):
        # accs / losses
        self.losses_train = np.zeros(p.num_iters) # training loss curve (should be plotted against p.its)
        self.losses_val = np.zeros(p.num_iters)  # testing loss curve (should be plotted against p.its)
        self.accs_train = np.zeros(p.num_iters)   # training acc curve (should be plotted against p.its)               
        self.accs_val = np.zeros(p.num_iters)    # testing acc curve (should be plotted against p.its)
        self.model_weights = None
        self.comp_model_weights = None
        self.explanation_divergence = np.zeros(p.num_iters)
        self.decoy_strength = 0
        self.use_individual = True
        self.test_acc = -1
        self.test_loss = -1
    
    # dictionary of everything but weights
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()}