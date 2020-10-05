import itertools
from slurmpy import Slurm
import os

params_to_vary = {

    'signal_strength': [100* (x) for x in range(4)]  , 
    'seed': [0,],
}

ks = sorted(params_to_vary.keys())

vals = [params_to_vary[k] for k in ks]

param_combinations = list(itertools.product(*vals)) # list of tuples

functions = [ "python train_with_decoy.py ",  ]# "python train_biased_gender.py ","python train_biased.py ",

for param_str in functions:
    for i in range(len(param_combinations)):
        cur_function = param_str

        for j, key in enumerate(ks):
            cur_function += '--'+key + ' ' + str(param_combinations[i][j]) + ' '

        os.system(cur_function)
        