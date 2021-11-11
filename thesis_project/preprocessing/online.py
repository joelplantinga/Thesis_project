import numpy as np
import pandas as pd

from river.preprocessing import OneHotEncoder
from river.compose import Select
from river.base import Transformer


class MultiLabelEncoder(Transformer):
    
    def __init__(self, features):
        
        self.features = features
        self.encoder = {}

        for feature in features:
            self.encoder[feature] = []

    def learn_one(self, x):

        for feat in self.features:
            if not x[feat] in self.encoder[feat]:
                self.encoder[feat].append(x[feat])
        
        return self

    def transform_one(self, x):

        for feat in self.features:
            x[feat] = self.encoder[feat].index(x[feat])
        
        return x

class MultiOneHotEncoder(Transformer):

    def __init__(self, features):

        self.encoder = Select(*features) | OneHotEncoder()

    def learn_one(self, x):

        self.encoder.learn_one(x)        
        return self

    def transform_one(self, x):

        x = self.encoder.transform_one(x)
        return x

class NoEncoder(Transformer):

    def __init__(self, features):

        self.features = features

    def learn_one(self, x):

        return self

    def transform_one(self, x):

        for feat in self.features:
            del x[feat]
        
        return x

