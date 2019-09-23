import itertools
from slurmpy import Slurm
import os
#partition = 'low'

# sweep different ways to initialize weights
params_to_vary = {
    
    'seed': [x for x in range(5)],
    
    'regularizer_rate': [1, 10, 100, 1000, 10000],
    'grad_method': [0,1,2]
    
}
 
ks = [x for x in params_to_vary.keys()]
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)
#for param_delete in params_to_delete:
#    param_combinations.remove(param_delete)

# iterate
import os

for i in range(len(param_combinations)):
    param_str = 'python train_mnist_decoy.py '
    for j, key in enumerate(ks):
        param_str += '--'+key + ' ' + str(param_combinations[i][j]) + ' '
    #s.run(param_str)
    print(param_str)
    os.system(param_str)
    