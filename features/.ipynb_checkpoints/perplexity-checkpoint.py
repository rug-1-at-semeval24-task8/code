import numpy as np
import torch
import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPT2TokenizerFast)

from features.base import FeatureGenerator

FIXED_LEN = 64
BATCH_SIZE = 16

class PerplexityFeature(FeatureGenerator):
    def __init__(self, device, model_id):
        self.device = device
        self.model_id = "gpt2"
        
    def features(self, documents):
        results = np.zeros((len(documents), 1)) 
        tokenizer = GPT2LMHeadModel.from_pretrained(self.model_id)
        model = GPT2TokenizerFast.from_pretrained(self.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.to(self.device)

        batches = [documents[i:i + BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]
        progress_bar = tqdm(range(len(batches)), ascii=True)
        with torch.no_grad():
            for i_b, batch in enumerate(batches):
                encodings = tokenizer(batch, return_tensors="pt")
                loss = model(input_ids=encodings["input_ids"], labels=encodings["input_ids"]).loss
        
                ppl = torch.exp(loss)
                ppl = ppl.item()
                for i in range(len(batch)):
                    results[i_b * BATCH_SIZE + i] = ppl
                progress_bar.update(1)
        return results