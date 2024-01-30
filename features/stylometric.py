from features.base import FeatureGenerator
from stanza.pipeline.multilingual import MultilingualPipeline
import numpy as np
from features.utils import timeit
from tqdm import tqdm

langs = ['ar','ru','zh-hans','id','ur','bg','de','en']
lang_configs = {lang: {"processors": "tokenize"} for lang in langs}
puncts = ".,?!:;-()"

class StylometricFeatures(FeatureGenerator):
    def __init__(self):
        self.nlp = MultilingualPipeline(lang_id_config={"langid_lang_subset": langs}, lang_configs=lang_configs)

    @timeit
    def features(self, documents):
        print("Computing StylometricFeatures")
        results = np.zeros((len(documents), 3))
        for i_doc, doc in enumerate(tqdm(documents)):
            processed = self.nlp(doc)
            sentences = processed.sentences
            tokens = [tok.text for sent in sentences for tok in sent.tokens]
            results[i_doc] = [
                # count capitalized words
                sum([1 for tok in tokens if tok[0].isupper()]) / len(tokens),
                # count punctuation
                sum([1 for tok in tokens if tok in puncts]) / len(tokens),
                # average sentence length
                np.mean([len(sent.tokens) for sent in sentences])
            ]
        return results