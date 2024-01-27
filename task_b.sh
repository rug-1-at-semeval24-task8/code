#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=taskb
#SBATCH --mem=40G
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/venvs/semeval/bin/activate

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'


python3.9 main.py --train_file_path ../SemEval2024-task8/data/SubtaskB/subtaskB_train.jsonl \
	--test_file_path ../SemEval2024-task8/data/SubtaskB/subtaskB_dev.jsonl \
	--prediction_file_path out/task_b_out --subtask B  --tags task_b $* 

