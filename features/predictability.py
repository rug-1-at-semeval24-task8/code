import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from features.base import FeatureGenerator

FIXED_LEN = 64
BATCH_SIZE = 16
eps = 1e-40

class PredictabilityFeature(FeatureGenerator):
    def __init__(self, device, local_device, language):
        self.device = device
        self.local_device = local_device
        self.language = language
        if language == 'en':
            self.models = ["gpt2"]
        else:
            raise NotImplementedError("Language not supported")

    def features(self, sentences):
        FEATURE_NUM = 3
        results = np.zeros((len(sentences), FIXED_LEN, FEATURE_NUM * len(self.models) + 1))
        reference_tokenisations = {}
        aggregated_results = np.zeros((len(sentences), FEATURE_NUM * len(self.models)))
        for i_m, model_id in enumerate(self.models):
            print("Computing perplexity using " + model_id)
            # Can we use the same transformer API for all models types?
            model = AutoModelForCausalLM.from_pretrained(model_id, is_decoder=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.to(self.device)


            batches = [sentences[i:i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
            progress_bar = tqdm(range(len(batches)), ascii=True)
            with torch.no_grad():
                for i_b, batch in enumerate(batches):
                    encodings = tokenizer(batch, padding=True, truncation=True, max_length=FIXED_LEN,
                                          return_offsets_mapping=True, return_tensors="pt")
                    encodings['input_ids'][encodings['input_ids'] >= 1000] = 0
                    if i_m == 0:
                        reference_tokenisations[i_b] = encodings['offset_mapping'].numpy()
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    target_ids = encodings['input_ids'].clone()
                    outputs = model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
                    logits = outputs['logits']
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()
                    probs = torch.nn.functional.softmax(shift_logits, dim=-1)
                    probs_seen = torch.gather(probs, 2, shift_labels.reshape(shift_labels.shape + (1,))).reshape(
                        shift_labels.shape)
                    greedy = torch.argmax(probs, -1)
                    probs_greedy = torch.gather(probs, 2, greedy.reshape(greedy.shape + (1,))).reshape(
                        greedy.shape)
                    # Prepare the vectors for output
                    log_probs_seen = torch.log(probs_seen).to(self.local_device).numpy()
                    log_probs_seen = np.concatenate((np.array([[0]] * len(batch)), log_probs_seen), axis=1)
                    log_probs_greedy = torch.log(probs_greedy).to(self.local_device).numpy()
                    log_probs_greedy = np.concatenate((np.array([[0]] * len(batch)), log_probs_greedy), axis=1)
                    entropy = torch.sum(torch.log2(probs + eps) * (-probs), dim=-1).to(self.local_device).numpy()
                    entropy = np.concatenate((np.array([[0]] * len(batch)), entropy), axis=1)
                    # Store the mask data
                    mask = np.array([x.to(self.local_device).numpy() for x in encodings['attention_mask']])
                    if i_m == 0:
                        results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1], 0] = mask
                    # Store the data
                    # Output per-token results
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1],
                    i_m * FEATURE_NUM + 1] = log_probs_seen * mask
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1],
                    i_m * FEATURE_NUM + 2] = log_probs_greedy * mask
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1],
                    i_m * FEATURE_NUM + 3] = entropy * mask
                    # Output aggregated results
                    for i in range(len(batch)):
                        aggregated_results[i_b * BATCH_SIZE + i, i_m * FEATURE_NUM] = np.mean(
                            log_probs_seen[i, mask[i] == 1])
                        aggregated_results[i_b * BATCH_SIZE + i, i_m * FEATURE_NUM + 1] = np.mean(
                            log_probs_greedy[i, mask[i] == 1])
                        aggregated_results[i_b * BATCH_SIZE + i, i_m * FEATURE_NUM + 2] = np.mean(
                            entropy[i, mask[i] == 1])
                    progress_bar.update(1)
        return results