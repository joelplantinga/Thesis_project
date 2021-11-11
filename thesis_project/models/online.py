from datetime import datetime
import pandas as pd
from river import ensemble, stream, metrics, preprocessing as pp, compose
import river
from time import perf_counter
from sklearn import metrics as sk_metrics
from random import random
import pickle
from preprocessing.online import MultiLabelEncoder, MultiOneHotEncoder, NoEncoder
import easygui as g

class Online:

    def __init__(self, encoding, feature_engineering=True, hyper_params=None):

        self.encoder = self.__init_encoding(encoding)
        self.RF = self.__init_RF(hyper_params)
        self.feature_engineering = feature_engineering

    def __init_encoding(self, encoding):

        if(encoding == 'label'):
            encoder = MultiLabelEncoder(
                ['prompt_type', 'device', 'prompt_description'])
        if(encoding == 'ohe'):
            encoder = MultiOneHotEncoder(
                ['prompt_type', 'device', 'user_id', 'prompt_description'])
        if(encoding == 'none'):
            encoder = NoEncoder(
                ['prompt_type', 'device', 'prompt_description'])

        return encoder

    def __calc_dist(self, x):
        x['dist'] = round(pow(pow(x['room_x'] - x['user_x'], 2) + 
                          pow(x['room_y'] - x['user_y'], 2), 0.5),
                          2)

        del x['user_x']
        del x['user_y']
        del x['room_x']
        del x['room_y']

        return x

    def __calc_floor_dif(self, x):
        
        x['floor_div'] = abs(x['user_floor'] - x['room_floor'])

        del x['user_floor']
        del x['room_floor']

        return x

    def __date_mod(self, x):
        
        date = pd.to_datetime(x["date_time"], format='%Y-%m-%d %H:%M:%S')

        x["diff_in_time"] = (date - datetime.strptime("1970-01-01", "%Y-%m-%d")).days
        x["month"] = date.month
        del x["date_time"]
        return x

    def prepare_data(self, x):

        if(self.feature_engineering):

            # Make sure that the features are understood well by the model.
            x = self.__calc_dist(x)
            x = self.__calc_floor_dif(x)
            x = self.__date_mod(x)

            self.encoder.learn_one(x)
            x = self.encoder.transform_one(x)

        else:
            for feat in ['date_time', 'prompt_type', 'prompt_description', 'device']:
                del x[feat]

        return x

    # TODO: Needs revision. Make this function more neat.
    def __init_RF(self, hyper_params):

        if hyper_params is None:
            model = ensemble.AdaptiveRandomForestClassifier(
                n_models=70,
                seed=42
            )
        else:
            model = ensemble.AdaptiveRandomForestClassifier(
                n_models=hyper_params['n_models'],
                max_features=hyper_params['max_features'],
                max_depth=hyper_params['max_depth']
            )

        return model
    
    def predict_one(self, x, class_pred, pp_first=True):

        if pp_first:
            x = self.prepare_data(x)

        # predict outcome and update the model
        if(class_pred):
            y_pred = self.RF.predict_one(x)
        else:
            y_pred = self.RF.predict_proba_one(x)

            if len(y_pred) == 1: # make sure it will be a tuple. 
                y_pred[not next(iter(y_pred))] = 0

        # TODO: In the batch model we did not use random for no data but instead kept it None.
        if(y_pred == None):
            y_pred = (random() > 0.5)
        
        return y_pred, x

    def train_predict_many(self, X, Y, class_pred):

        y_pred_list = []

        # As the model works online, the data is accessed per record.
        for xi, yi in stream.iter_pandas(X, Y):

            y_pred, xi = self.predict_one(xi, class_pred)

            if not class_pred:
                y_pred = y_pred[True]
            
            y_pred_list.append(y_pred)
            
            self.RF.learn_one(xi, yi)

        return y_pred_list

    def online_model(self, data_blocks, class_pred=True, print_freq=1):
        
        test_res = {"accuracy":[], "size":[], "time":[]}

        for i, block in enumerate(data_blocks):
            
            time = perf_counter()
            
            y_true = block.pop('classification')

            if(not class_pred):
                weights = block.pop('total_weight')
                block = block.drop(['user_weight', 'prompt_weight'], axis=1)

            y_pred = self.train_predict_many(block, y_true, class_pred)
            
            time_measured = perf_counter() - time

            if class_pred:
                accuracy = sk_metrics.accuracy_score(y_true, y_pred) * 100
            else:
                accuracy = sk_metrics.mean_absolute_error(weights, y_pred)

            test_res['accuracy'].append(accuracy)
            test_res['size'].append(len(block))
            test_res['time'].append(time_measured)

            if(isinstance(print_freq, int) and i % print_freq == 0):
                print("#", i, "(size:", len(block), ")",  "acc:", round(accuracy, 3), 
                        "time:", round(time_measured, 3))
            
        return test_res

    def save_model(self, filename):

        path = g.diropenbox()

        with open(path+"/"+filename+".pickle", "wb") as output_file:
            pickle.dump(self, output_file)
