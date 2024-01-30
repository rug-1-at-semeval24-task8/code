from features.base import FeatureGenerator
from stanza.pipeline.multilingual import MultilingualPipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from features.utils import timeit
from tqdm import tqdm

langs = ['ar','ru','zh-hans','id','ur','bg','de','en']
lang_configs = {lang: {"processors": "tokenize,mwt,pos"} for lang in langs}
pos_vocab = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

class POSFeature(FeatureGenerator):
    def __init__(self):
        self.nlp = MultilingualPipeline(lang_id_config={"langid_lang_subset": langs}, lang_configs=lang_configs)
        self.vect = CountVectorizer(vocabulary=pos_vocab, lowercase=False)

    @timeit
    def features(self, documents):
        print("Computing POSFeature")
        results = np.zeros((len(documents), len(pos_vocab)))
        for i_doc, doc in enumerate(tqdm(documents)):
            processed = self.nlp(doc)
            pos = [word.upos for sent in processed.sentences for word in sent.words]
            mat = self.vect.transform([' '.join(pos)])
            freq = mat / mat.sum()
            results[i_doc] = freq.toarray()
        return results
