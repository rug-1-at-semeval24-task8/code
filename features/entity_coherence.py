from features.base import FeatureGenerator
from stanza import Pipeline
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from features.utils import timeit

role_rank = {
  "nsubj": 3,
  "nsubjpass": 3,
  "csubj": 3,
  "csubjpass": 3,
  "agent": 3,
  "obj": 2,
  "dobj": 2,
  "iobj": 2,
  "pobj": 2,
  "dative": 2
}
role_map = {
  "nsubj": "S",
  "nsubjpass": "S",
  "csubj": "S",
  "csubjpass": "S",
  "agent": "S",
  "obj": "O",
  "dobj": "O",
  "iobj": "O",
  "pobj": "O",
  "dative": "O",
  "-": "-"
}
roles = ["S","O","X","-"]
possible_transitions = [a + b for a in roles for b in roles]

class EntityCoherenceFeature(FeatureGenerator):
    def __init__(self, device, local_device):
        self.device = device
        self.local_device = local_device
        self.nlp = Pipeline("en", processors="tokenize,mwt,pos,lemma,depparse,coref")
        self.vect = CountVectorizer(vocabulary=possible_transitions, ngram_range=(2,2), analyzer="char", lowercase=False)

    @timeit
    def features(self, documents):
        results = np.zeros((len(documents), 16)) 
        for i_doc, doc in enumerate(documents):
            # split document into sentences
            processed = self.nlp(doc)
            chains = processed.coref
            # get mentions for each referent in each sentence
            mentions_by_sentence = [[
                [m.start_word for m in c.mentions if m.sentence == s]
                for s in range(len(processed.sentences))]
                for c in chains]
            # get the dependency roles for the mentions
            ## for multiple mentions per sentence, get the 'highest' role
            ### (subjects > objects > others)
            ## for sentences with no mentions, assign '-'
            roles_by_sentence = [
                [max([processed.sentences[i].words[m].deprel for m in s],
                     key=lambda role: role_rank.get(role, 1))
                     if len(s) > 0 else "-" for i, s in enumerate(c)]
                for c in mentions_by_sentence]
            # transform role arrays into sequences, and vectorize
            role_seq_by_sentence = ["".join([role_map.get(s, "X") for s in c]) for c in roles_by_sentence]
            mat = self.vect.transform(role_seq_by_sentence)
            dist = mat.sum(axis=0) / mat.sum()
            results[i_doc] = dist.toarray()
        return results

