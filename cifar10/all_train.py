import itertools

import os

params_to_vary = {
    
    'seed': [x+10 for x in range(3)],
    
    'regularizer_rate': [1.0, ],
    'grad_method': [0 ,], #set to [0,1,2] to calculate for all methods
    
    'epochs': [100 ,], #set to [0,1,2] to calculate for all methods

    'batch_size': [256,] #set to [0,1,2] to calculate for all methods
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
    param_str = 'python train.py '
    for j, key in enumerate(ks):
        param_str += '--'+key + ' ' + str(param_combinations[i][j]) + ' '
    #s.run(param_str)
    print(param_str)
    os.system(param_str)
    
    
    
