#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=taska_mono
#SBATCH --mem=40G
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/venvs/semeval/bin/activate

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3.9 main.py --train_file_path ../SemEval2024-task8/data/SubtaskA/subtaskA_train_monolingual.jsonl \
	--test_file_path data.jsonl \
	--prediction_file_path out/task_a_mono_out --subtask A --tags task_a_mono --mono $* 




