from features.base import FeatureGenerator
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from features.utils import timeit

class TfidfFeatures(FeatureGenerator):
    def __init__(self, vectorizer):
        self.vect = load(vectorizer)

    @timeit
    def features(self, documents):
        print("Computing TfidfFeatures")
        results = self.vect.transform(documents).toarray()
        return results