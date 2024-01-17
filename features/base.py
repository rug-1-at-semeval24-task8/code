from abc import ABC, abstractmethod
from features.utils import timeit

class FeatureGenerator(ABC):

    @timeit
    @abstractmethod
    def features(self, sentences):
        ...
    
