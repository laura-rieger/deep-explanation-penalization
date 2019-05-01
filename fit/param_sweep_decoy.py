import itertools
from slurmpy import Slurm
import os
#partition = 'low'

# sweep different ways to initialize weights= 
params_to_vary = {
    'signal_strength':   [100*(x) for x in range(10)]    ,
    'decoy_strength' : [ 100.0,]
}

ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)
#for param_delete in params_to_delete:
#    param_combinations.remove(param_delete)

# iterate
import os

for i in range(len(param_combinations)):
    param_str = 'python train_with_adversarial.py '
    for j, key in enumerate(ks):
        param_str += '--'+key + ' ' + str(param_combinations[i][j]) + ' '
    #s.run(param_str)
    print(param_str)
    os.system(param_str)
    