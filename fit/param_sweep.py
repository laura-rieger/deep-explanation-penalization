import itertools
from slurmpy import Slurm
import os
#partition = 'low'

# sweep different ways to initialize weights
params_to_vary = {
    'signal_strength': [1.0],
    'train_both': [0, 1],
    'sparse_signal': [0, 1]
}
CUDA_DEVICE=0
# run
#s = Slurm("sweep_full", {"partition": partition, "time": "3-0"})
ks = sorted(params_to_vary.keys())
vals = [params_to_vary[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples
print(param_combinations)
# for param_delete in params_to_delete:
#     param_combinations.remove(param_delete)

# iterate
for i in range(len(param_combinations)):
    param_str = 'CUDA_VISIBLE_DEVICES=' + str(CUDA_DEVICE)+ ' python train.py '
    for j, key in enumerate(ks):
        param_str += '--'+key + ' ' + str(param_combinations[i][j]) + ' '
    #s.run(param_str)
    print(param_str)
    os.system(param_str)
    