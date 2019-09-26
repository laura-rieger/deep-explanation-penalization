import itertools
from slurmpy import Slurm
import os

params_to_vary = {

    'signal_strength': [0, 100, 200, 300 , 400, 500, 600]  ,
    'seed': [x for x in range(5)],
}

ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)

# functions = ["python train_biased.py ", "python train_biased_gender.py ", "python train_with_decoy.py ",  ]
functions = ["python train_biased_gender.py ",  "python train_biased.py ",  ]
for param_str in functions:
    for i in range(len(param_combinations)):
        my_func = param_str
    # change script here
        #param_str = 'python train_biased_gender.py '
        for j, key in enumerate(ks):
            my_func += '--'+key + ' ' + str(param_combinations[i][j]) + ' '
        print(my_func)
        os.system(my_func)
        