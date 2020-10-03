import itertools
from slurmpy import Slurm
import os


params_to_vary = {
    'regularizer_rate':   [0,10, -1]    ,
     'seed':   [ x for x in range(5)]    , 
     

}

ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)


for i in range(len(param_combinations)):
    param_str = 'python train_saliency.py '
    for j, key in enumerate(ks):
        param_str += '--'+key + ' ' + str(param_combinations[i][j]) + ' '

    os.system(param_str)
    