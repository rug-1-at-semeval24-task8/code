from features.base import FeatureGenerator
from stanza.pipeline.multilingual import MultilingualPipeline
import torch
import numpy as np
from sklearn.feature_extraction import CountVectorizer

langs = ['ar','ru','zh-hans','id','ur','bg','de','en']
lang_configs = {lang: {"processors": "tokenize"} for lang in langs}

class InformationRedundancyFeatures(FeatureGenerator):
    def __init__(self, device, local_device, batch_size):
        self.device = device
        self.local_device = local_device
        self.batch_size = batch_size
        self.nlp = MultilingualPipeline(lang_id_config={"langid_lang_subset": langs}, lang_configs=lang_configs)
        self.trunc_frac = 0.25

    def features(self, documents):
        results = np.zeros((len(documents), 6)) 
        for i_doc, doc in enumerate(documents):
            # split document into sentences
            processed = self.nlp(doc)
            sentences = [s.text.strip() for s in processed.sentences]
            # create sentence-term matrix
            vect = CountVectorizer()
            mat = vect.fit_transform(sentences)
            sentence_term_matrix = torch.from_numpy(mat.toarray()).float()
            # create sentence-sentence matrix
            original = torch.mm(torch.transpose(sentence_term_matrix, 0, 1), sentence_term_matrix)
            # truncate sentence-sentence matrix
            u, s, v = torch.svd_lowrank(original, q=min(6, *original.shape[-2:]))
            n = round(self.trunc_frac * s.shape[0])
            s[-n:] = 0
            truncated = u.mm(torch.diag_embed(s)).mm(v.transpose(0, 1))
            # calculate matrix difference between original and truncated
            difference = torch.sub(original, truncated)
            results[i_doc] = [
                # descriptive stats of difference: norm and sum
                torch.linalg.matrix_norm(difference).pow(2).item(),
                torch.sum(difference).item(),
                # descriptive stats of truncated matrix
                torch.max(truncated).item(),
                torch.min(truncated).item(),
                torch.mean(truncated).item(),
                torch.median(truncated).item()
            ]
        return results

