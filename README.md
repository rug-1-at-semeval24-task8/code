# How-to

* Python version **3.9.6**.

* Install requirements:
```sh
pip install -r requirements.txt
```

* Example run:

```sh
python --train_file_path ../SemEval2024-task8/data/SubtaskA/subtaskA_train_multilingual.jsonl --test_file_path ../SemEval2024-task8/data/SubtaskA/subtaskA_dev_multilingual.jsonl --prediction_file_pat task_a_multi_out --subtask A --tags task_a_multi -data_size 100
```