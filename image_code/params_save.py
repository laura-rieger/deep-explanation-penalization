import numpy as np
from numpy.random import randint

class S:   
    def __init__(self, num_epochs):
        # accs / losses
        self.losses_train = np.zeros(num_epochs) # training loss curve (should be plotted against p.its)
        self.losses_test = np.zeros(num_epochs)  # testing loss curve (should be plotted against p.its)
        self.accs_train = np.zeros(num_epochs)   # training acc curve (should be plotted against p.its)               
        self.accs_test = np.zeros(num_epochs)    # testing acc curve (should be plotted against p.its)
                    
        self.cd = np.zeros(num_epochs)    # testing acc curve (should be plotted against p.its)
        self.model_weights = None
        self.regularizer_rate = 0
        self.blocks = True
        self.num_blobs = 0
        self.seed = 42

    
    # dictionary of everything but weights
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()}