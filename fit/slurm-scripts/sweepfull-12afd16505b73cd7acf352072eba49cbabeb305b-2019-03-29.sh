#!/bin/bash

#SBATCH -e logs/sweepfull-12afd16505b73cd7acf352072eba49cbabeb305b-2019-03-29.%J.err
#SBATCH -o logs/sweepfull-12afd16505b73cd7acf352072eba49cbabeb305b-2019-03-29.%J.out
#SBATCH -J sweepfull-12afd16505b73cd7acf352072eba49cbabeb305b-2019-03-29

#SBATCH --partition=low
#SBATCH --time=3-0

set -eo pipefail -o nounset


###
train.py --signal_strength 1.0 --sparse_signal 0 --train_both 0 