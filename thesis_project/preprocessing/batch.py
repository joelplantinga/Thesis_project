import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import FeatureHasher


class MultiLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, features):

        self.features = features
        self.encoders = {}

        for feature in features:
            self.encoders[feature] = LabelEncoder()

    def fit_transform(self, X, y=None, **fit_params):

        for ft in self.features:

            if not ft in X.columns:
                print("Trying to access a column in the label encoder that not exists.")
                continue

            X[ft] = self.encoders[ft].fit_transform(X[ft])
        
        return X

class MultiOneHotEncoder(BaseEstimator, TransformerMixin):

        def __init__(self, features):

            self.features = features
            self.start = True
            self.OHE = OneHotEncoder(handle_unknown='ignore')

        def fit_transform(self, X, y=None, **fit_params):

            enc_data = np.array(X[self.features])

            if self.start:
                enc_data = self.OHE.fit_transform(enc_data).toarray()
                self.start = False
            
            else:
                enc_data = self.OHE.transform(enc_data).toarray()

            enc_data = pd.DataFrame(enc_data, columns=self.OHE.get_feature_names())
            X = X.drop(self.features, axis=1)
            X = X.reset_index()
            X = pd.concat([X, enc_data], axis=1)

            return X

class NoEncoder(BaseEstimator, TransformerMixin):

        def __init__(self, features):

            self.features = features

        def fit_transform(self, X, y=None, **fit_params):

            X = X.drop(self.features, axis=1)

            return X

class MultiFeatureHasher(BaseEstimator, TransformerMixin):

    def __init__(self, features):

        self.features = features
        self.encoders = {}

        for feature, n_feat in features:
            self.encoders[feature] = FeatureHasher(n_features=n_feat, input_type='string')
    
    def fit_transform(self, X, y=None, **fit_params):

        features = [ft[0] for ft in self.features]
        
        for feature in features:

            X[feature] = X[feature].apply(str)

            hashed_features = self.encoders[feature].fit_transform(X[feature])

            hashed_features = pd.DataFrame(hashed_features.toarray())
            hashed_features = hashed_features.add_prefix(feature + "_")

            X = pd.concat([X.reset_index(drop=True), hashed_features], axis=1)

        X = X.drop(features, axis=1)
        
        return X
            
        
