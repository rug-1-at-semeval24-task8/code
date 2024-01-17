import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from features.utils import timeit

from features.base import FeatureGenerator

# Here is an example document level feature generator.
# It is a feature generator that computes the perplexity of a document
# using a pretrained model.
# Document level feature generators should inherit from FeatureGenerator
# and implement the features method.
# The features method should return a numpy array of shape (len(documents), num_features)
# where num_features is the number of features generated per document.
# In this case, we are generating one feature per document, so num_features = 1.
# The documents argument is a list of documents, where each document is a string.
# The return value is a numpy array of shape (len(documents), 1) where each entry is the perplexity of the corresponding document.
# The perplexity is computed using a pretrained model.

# During the call to the features method, your feature generator should take the all the documents and compute the features for each document.
# Please have a look at the code in main.py to see how the feature generators are used.


class PerplexityFeature(FeatureGenerator):
    def __init__(self, device, local_device, model_id, fixed_length, experiment):
        self.device = device
        self.fixed_length = fixed_length
        self.model_id = model_id
        self.local_device = local_device
        self.batch_size = 1
        self.experiment = experiment

    @timeit
    def features(self, documents):
        print("Computing PerplexityFeature")
        results = np.zeros((len(documents), 1))
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.to(self.device)

        batches = [
            documents[i : i + self.batch_size]
            for i in range(0, len(documents), self.batch_size)
        ]
        progress_bar = tqdm(range(len(batches)), ascii=True)
        with torch.no_grad():
            for i_b, batch in enumerate(batches):
                encodings = tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=self.fixed_length,
                    truncation=True,
                    padding=True,
                ).to(self.device)
                loss = model(
                    input_ids=encodings["input_ids"], labels=encodings["input_ids"]
                ).loss

                ppl = torch.exp(loss)
                ppl = ppl.item()
                for i in range(len(batch)):
                    results[i_b * self.batch_size + i] = ppl
                progress_bar.update(1)
        return results
