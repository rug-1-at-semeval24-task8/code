#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=a100.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=taska_mono
#SBATCH --mem=40G
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/venvs/semeval/bin/activate


python3.9 main.py --train_file_path ../SemEval2024-task8/data/SubtaskA/subtaskA_train_monolingual.jsonl \
	--test_file_path ../SemEval2024-task8/data/SubtaskA/subtaskA_dev_monolingual.jsonl \
	--prediction_file_path test_out --subtask A --model A --data_size 128 --tags test $*




