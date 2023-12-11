import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from features.base import FeatureGenerator

FIXED_LEN = 64

class PerplexityFeature(FeatureGenerator):
    def __init__(self, device, local_device, model_id, batch_size):
        self.device = device
        self.model_id = model_id
        self.local_device = local_device
        self.batch_size = batch_size
        
    def features(self, documents):
        results = np.zeros((len(documents), 1)) 
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.to(self.device)

        batches = [documents[i:i + self.batch_size] for i in range(0, len(documents), self.batch_size)]
        progress_bar = tqdm(range(len(batches)), ascii=True)
        with torch.no_grad():
            for i_b, batch in enumerate(batches):
                encodings = tokenizer(batch, return_tensors="pt",  max_length=FIXED_LEN, truncation=True, padding=True)
                loss = model(input_ids=encodings["input_ids"], labels=encodings["input_ids"]).loss
            
                ppl = torch.exp(loss)
                ppl = ppl.item()
                for i in range(len(batch)):
                    results[i_b * self.batch_size + i] = ppl
                progress_bar.update(1)
        return results