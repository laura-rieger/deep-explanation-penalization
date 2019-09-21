import numpy as np
from numpy.random import randint

class S:   
    def __init__(self, num_epochs):
        # accs / losses
        self.losses_train = []# training loss curve (should be plotted against p.its)
        self.losses_test = []  # testing loss curve (should be plotted against p.its)
        self.accs_train = []   # training acc curve (should be plotted against p.its)               
        self.accs_test = []    # testing acc curve (should be plotted against p.its)
                    
        self.cd = []    # testing acc curve (should be plotted against p.its)
        self.model_weights = None
        self.regularizer_rate = 0
        self.blocks = True
        self.dataset= "Color"
        self.method = "CD"
        self.num_blobs = 0
        self.seed = 42

    
    # dictionary of everything but weights
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()}