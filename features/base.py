from abc import ABC, abstractmethod

class FeatureGenerator(ABC):

    @abstractmethod
    def features(self, sentences):
        ...
    
