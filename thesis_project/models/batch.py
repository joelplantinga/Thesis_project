from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction import FeatureHasher

import preprocessing.batch as pp #import MultiLabelEncoder, MultiOneHotEncoder, NoEncoder
import easygui as g
import pickle


class Batch():

    def __init__(self, encoding, hyper_params=None, feature_engineering=True):

        self.feature_engineering = feature_engineering
        self.encoder = self.__init_encoding(encoding)
        self.RF = self.__init_RF(hyper_params, encoding)
        self.feature_imp = None

    def __init_encoding(self, encoding):

        if(encoding == 'ohe'):
            encoder = pp.MultiOneHotEncoder(['prompt_type', 'device', 'user_id', 'prompt_description'])
        elif(encoding == 'label'):
            encoder = pp.MultiLabelEncoder(['prompt_type', 'device', 'prompt_description'])
        elif(encoding == 'hash'):
            encoder = pp.MultiFeatureHasher([('prompt_type', 2), ('device', 8),
                                            ('user_id', 100), ('prompt_description', 25)])
        elif(encoding == 'none'):
            encoder = pp.NoEncoder(['prompt_type', 'device', 'prompt_description'])

        return encoder

    # TODO: Make sure the max_features/x_cols works correctly.
    # NOTE: max cols protection is removed
    def __init_RF(self, params, encoding, x_cols=20):
        
        if(encoding == 'ohe' and self.feature_engineering):
            params = {"n_estimators": 300, "max_depth": 30, 
                      "min_samples_split": 5, "min_samples_leaf": 6,
                       "max_features":'auto'}
        elif(params == "label" and self.feature_engineering):
            params = {"n_estimators": 255, "max_depth": 5, 
                      "min_samples_split": 2, "min_samples_leaf": 1,
                       "max_features":'auto'}
        elif(params == "none" and self.feature_engineering):
            params = {"n_estimators": 500, "max_depth": 10, 
                      "min_samples_split": 10, "min_samples_leaf": 5,
                       "max_features":'auto'}
        elif(params == "no_ft"):
            params = {"n_estimators": 400, "max_depth": 10, 
                      "min_samples_split": 2, "min_samples_leaf": 6,
                       "max_features":'auto'}
        elif(params==None):
            params = {"n_estimators": 100, "max_depth": None, 
                      "min_samples_split": 2, "min_samples_leaf": 1,
                       "max_features":'auto'}


        forest = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                        max_depth=params["max_depth"], 
                                        min_samples_split=params["min_samples_split"],
                                        min_samples_leaf=params["min_samples_leaf"], 
                                        max_features=params["max_features"])
        return forest

    def __calc_dist(self, df):
        
        """Function that calculates the distance between the user
        and the prompt. 

        Attributes:
        ------------

        df : pd.Dataframe
            Dataset that contains the user and prompt data.

        Returns:
        ------------

        df : pd.Dataframe
            Dataset including distance between prompt and user. Without
            coordinates.

        """
        df.loc[:,'dist'] = round(pow(pow(df['room_x'] - df['user_x'], 2) + 
                           pow(df['room_y'] - df['user_y'], 2), 0.5),
                           2)

        del df['user_x']
        del df['user_y']
        del df['room_x']
        del df['room_y']

        return df

    def __calc_floor_dif(self, df):
        
        """Function that calculates the difference between the floor
        of the user and the prompt. 

        Attributes:
        ------------

        data : pd.Dataframe
            Dataset that contains the user and prompt data.

        Returns:
        ------------

        data : pd.Dataframe
            Dataset including floor difference. Without
            floor numbers.

        """

        df.loc[:,'floor_div'] = abs(df['user_floor'] - df['room_floor'])

        del df['user_floor']
        del df['room_floor']

        return df

    def __date_mod(self, df):
        """Function that calculates the date as x_days from the unix date and
        the number of the month.

        Attributes:
        ------------

        df : pd.Dataframe
            Dataset that contains the user and prompt data.

        Returns:
        ------------

        df : pd.Dataframe
            Dataset containing modified features about the date. 

        """

        date = pd.to_datetime(df["date_time"], format='%Y-%m-%d %H:%M:%S')
        unix = datetime.strptime("1970-01-01", "%Y-%m-%d")

        df.loc[:,"diff_in_time"] = [ (dt - unix).days for dt in date ]

        df.loc[:,"month_diff"] = [ abs(6 - min(dt.month, 12-dt.month)) for dt in date ] 
        
        del df["date_time"]

        return df

    def prepare_data(self, df):
        
        if(self.feature_engineering):

            # delete correlated features
            # df = df.drop(['team', 'team_prompts', 'has_location'], axis=1)

            df = self.__calc_dist(df.copy())        
            df = self.__calc_floor_dif(df.copy())   
            df = self.__date_mod(df.copy()) 
            # df = df.drop(['date_time'], axis=1)

            df = self.encoder.fit_transform(df)

        else:
            df = df.drop(['date_time', 'prompt_type', 'prompt_description', 'device'], axis=1)

        return df

    def train(self, train_x, train_y):
        
        if(train_x.empty or train_y.empty):
            print("RETURN: Cannot train model on empty dataset.")
            return None

        train_x = self.prepare_data(train_x)

        self.RF.fit(train_x,train_y)

        self.feature_imp = pd.Series(
            self.RF.feature_importances_, index=train_x.columns).sort_values(ascending=False)

        return self.RF
    
    def predict(self, test_x, class_pred):

        test_x = self.prepare_data(test_x)

        if(class_pred):
            pred_y = self.RF.predict(test_x)
        else:
            pred_y = self.RF.predict_proba(test_x)

        return pred_y

    def train_predict(self, train_x, train_y, test_x, class_pred):
        
        check = self.train(train_x, train_y)

        if check is None:
            return None

        pred_y = self.predict(test_x, class_pred)

        return pred_y

    def batch_system(self, data, class_pred, print_freq=1):

        train_x = pd.DataFrame()
        train_y = pd.Series()

        test_res = {"accuracy":[], "size":[], "time":[]}

        for i, chunk in enumerate(data):

            time = perf_counter()

            test_y = chunk.pop("classification")

            if(not class_pred):
                weights = chunk.pop('total_weight')
                chunk = chunk.drop(['user_weight', 'prompt_weight'], axis=1)

            pred_y = self.train_predict(train_x.copy(), train_y.copy(), chunk.copy(), class_pred)

            train_x = train_x.append(chunk)
            train_y = train_y.append(test_y)

            time_measured = perf_counter() - time

            # The first time no prediction is made due to not having data. 
            # Therefore, no accuracy is measured.
            if(pred_y is None):
                continue
            
            if class_pred:
                accuracy = metrics.accuracy_score(test_y, pred_y) * 100
            else:
                pred_y = [item[1] for item in pred_y]
                accuracy = metrics.mean_absolute_error(weights, pred_y)

            test_res['time'].append(time_measured)
            test_res['accuracy'].append(accuracy)
            test_res['size'].append(len(chunk))

            if(isinstance(print_freq, int) and i % print_freq == 0):
                print("#", i, "(size:", len(test_y), ")",  "acc:", round(accuracy, 3), 
                      "time:", round(time_measured, 3))
        
        return test_res

    def test_model(self, data, test_size=0.5, class_pred=True):

        y = data.pop('classification')

        train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=test_size)

        pred_y = self.train_predict(train_x, train_y, test_x, class_pred)
        
        if class_pred:
            accuracy = metrics.accuracy_score(test_y, pred_y) * 100
        else:
            pred_y = [item[1] for item in pred_y]
            accuracy = metrics.mean_absolute_error(test_y, pred_y)

        return accuracy

    def test_prediction(self, data, class_pred=True):

        test_y = data.pop('classification')

        pred_y = self.predict(data, class_pred)
        
        if class_pred:
            accuracy = metrics.accuracy_score(test_y, pred_y) * 100
        else:
            pred_y = [item[1] for item in pred_y]
            accuracy = metrics.mean_absolute_error(test_y, pred_y)

        return accuracy

    def save_model(self, filename):

        path = g.diropenbox()

        with open(path+"/"+filename+".pickle", "wb") as output_file:
            pickle.dump(self, output_file)

