#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=jupyterlarge
#SBATCH --mem=40G
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/venvs/semeval/bin/activate

jupyter lab --no-browser --ip=0.0.0.0 --port=9998
