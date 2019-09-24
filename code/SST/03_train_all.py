import itertools
from slurmpy import Slurm
import os

params_to_vary = {

    'signal_strength': [400, 500, 600]  ,
    'seed': [x for x in range(5)],
}

ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)

functions = ["python train_biased.py ", "python train_biased_gender.py ", "python train_with_decoy.py ",  ]
for param_str in functions:
    for i in range(len(param_combinations)):
    # change script here
        #param_str = 'python train_biased_gender.py '
        for j, key in enumerate(ks):
            param_str += '--'+key + ' ' + str(param_combinations[i][j]) + ' '
        print(param_str)
        os.system(param_str)
        