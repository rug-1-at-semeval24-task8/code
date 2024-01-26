#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=taskb
#SBATCH --mem=40G
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/venvs/semeval/bin/activate


python3.9 main.py --train_file_path ../SemEval2024-task8/data/SubtaskB/subtaskB_train.jsonl \
	--test_file_path ../SemEval2024-task8/data/SubtaskB/subtaskB_dev.jsonl \
	--prediction_file_path task_b_out --subtask B  --tags task_b 

