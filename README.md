# How-to

* Python version **3.9.6**.

* Install requirements:
```sh
pip install -r requirements.txt
```
* Arguments
```
usage: main.py [-h] --train_file_path TRAIN_FILE_PATH --test_file_path
               TEST_FILE_PATH --subtask {A,B} [--models MODELS [MODELS ...]]
               --prediction_file_path PREDICTION_FILE_PATH
               [--feat_perplexity FEAT_PERPLEXITY] [--batch_size BATCH_SIZE]
               [--batch_size_feature_extraction BATCH_SIZE_FEATURE_EXTRACTION]
               [--epochs EPOCHS] [--fixed_length FIXED_LENGTH] [--seed SEED]
               [--enable_preditability | --no-enable_preditability | -ep]
               [--enable_perplexity | --no-enable_perplexity | -epp]
               [--enable_information_redundancy | --no-enable_information_redundancy | -eir]
               [--enable_entity_coherence | --no-enable_entity_coherence | -eec]
               [--data_size DATA_SIZE] [--tags TAGS [TAGS ...]]
               [--data_split_strategy DATA_SPLIT_STRATEGY [DATA_SPLIT_STRATEGY ...]]
               [--add_human_to_val | --no-add_human_to_val | -ahv]
               [--downsample_to_test_size | --no-downsample_to_test_size | -dts]
               [--hidden_size HIDDEN_SIZE] [--num_lstm_layers NUM_LSTM_LAYERS]
               [--dropout_prop DROPOUT_PROP]
               [--attention_enabled | --no-attention_enabled | -ae]

optional arguments:
  -h, --help            show this help message and exit
  --train_file_path TRAIN_FILE_PATH, -tr TRAIN_FILE_PATH
                        Path to the train file.
  --test_file_path TEST_FILE_PATH, -t TEST_FILE_PATH
                        Path to the test file.
  --subtask {A,B}, -sb {A,B}
                        Subtask (A or B).
  --models MODELS [MODELS ...], -m MODELS [MODELS ...]
                        Transformer to train and test
  --prediction_file_path PREDICTION_FILE_PATH, -p PREDICTION_FILE_PATH
                        Path where to save the prediction file.
  --feat_perplexity FEAT_PERPLEXITY, -fperp FEAT_PERPLEXITY
                        Feature perplexity
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size
  --batch_size_feature_extraction BATCH_SIZE_FEATURE_EXTRACTION, -bsfe BATCH_SIZE_FEATURE_EXTRACTION
                        Batch size
  --epochs EPOCHS, -e EPOCHS
                        Epochs
  --fixed_length FIXED_LENGTH, -fl FIXED_LENGTH
                        Fixed length
  --seed SEED, -s SEED  Seed
  --enable_preditability, --no-enable_preditability, -ep
                        Enable predictability feature
  --enable_perplexity, --no-enable_perplexity, -epp
                        Enable perplexity feature
  --enable_information_redundancy, --no-enable_information_redundancy, -eir
                        Enable information redundancy features
  --enable_entity_coherence, --no-enable_entity_coherence, -eec
                        Enable entity coherence features (note: this should
                        only be enabled for monolingual English data)
  --data_size DATA_SIZE, -ds DATA_SIZE
                        Data size
  --tags TAGS [TAGS ...], -tg TAGS [TAGS ...]
                        Tags
  --data_split_strategy DATA_SPLIT_STRATEGY [DATA_SPLIT_STRATEGY ...], -dss DATA_SPLIT_STRATEGY [DATA_SPLIT_STRATEGY ...]
                        Data split strategy
  --add_human_to_val, --no-add_human_to_val, -ahv
                        Add human to validation set (default: False)
  --downsample_to_test_size, --no-downsample_to_test_size, -dts
                        Downsample to test size to make it more comparable
                        with test data (default: False)
  --hidden_size HIDDEN_SIZE, -hs HIDDEN_SIZE
                        Hidden size
  --num_lstm_layers NUM_LSTM_LAYERS, -nl NUM_LSTM_LAYERS
                        Number of LSTM layers
  --dropout_prop DROPOUT_PROP, -dp DROPOUT_PROP
                        Dropout proportion
  --attention_enabled, --no-attention_enabled, -ae
                        Attention enabled (default: False)
```
* Example run:

```sh
python main.py --train_file_path ../SemEval2024-task8/data/SubtaskA/subtaskA_train_multilingual.jsonl --test_file_path ../SemEval2024-task8/data/SubtaskA/subtaskA_dev_multilingual.jsonl --prediction_file_pat task_a_multi_out --subtask A --tags task_a_multi --data_size 100 -ep -epp
```
