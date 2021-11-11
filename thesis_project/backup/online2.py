from datetime import datetime
import pandas as pd
from river import ensemble, stream, metrics, preprocessing as pp, compose
import river
from time import perf_counter
from sklearn import metrics as sk_metrics
from random import random


class Online:

    def __calc_dist(self, x):
        x['dist'] = round(pow(pow(x['room_x'] - x['user_x'], 2) + 
                          pow(x['room_y'] - x['user_y'], 2), 0.5),
                          2)

        del x['user_x']
        del x['user_y']
        del x['room_x']
        del x['room_y']

        return x

    def __init_encoder(self, features, encoding):
        
        if (encoding == 'label'):
            encoder = {}
            for feat in features:
                encoder[feat] = []

        elif (encoding == 'ohe'):
            # OneHotEncoding pipe. First the right features are chosen
            # before they are passed to the encoder.
            encoder = compose.Select(*features) | pp.OneHotEncoder()
        
        elif (encoding == 'none'):
            encoder = None

        return encoder

    def __label_encoder(self, encoder, x, features):

        for feat in features:
            if not(x[feat] in encoder[feat]):
                encoder[feat].append(x[feat])
            
            x[feat] = encoder[feat].index(x[feat])

        return x, encoder

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

    def drop_features(self, x, features):
        for f in features:
            del x[f]
        
        return x

    def basic(self, X):
        """Simplest online learning model that serves as comparison for other models."""

        model = ensemble.AdaptiveRandomForestClassifier( 
            n_models=3,
            seed=42
        )

        # The metric measures how good the models performs.
        metric = metrics.Precision()

        y = X.pop('classification')

        i = 0

        # As the model works online, the data is accessed per record.
        for xi, yi in stream.iter_pandas(X, y):
            
            # Delete categorical variables for now.
            del xi['date_time']
            del xi['prompt_type']
            del xi['prompt_description']
            del xi['device']


            # The model does the prediction first --
            y_pred = model.predict_one(xi)

            # -- And learns from that record directly afterwards.
            model.learn_one(xi, yi)

            # The first time, the model cannot predict and therefore returns
            # None. None however, is not accepted by update().
            if(y_pred != None):
                metric = metric.update(yi, y_pred)

            
            # For testing purposes.
            if (i % 5000 == 0):
                print("#", i, round(float(metric.get() * 100), 2))  
            i += 1

        return float(metric.get())

    def prepare_data(self, x, encoder, encoding, cat_variables):

        # Make sure that the features are understood well by the model.
        x = self.__calc_dist(x)     
        x = self.__calc_floor_dif(x)
        x = self.__date_mod(x)

        if (encoding == 'ohe'):
            # Memorise features and encode them into OHE features
            encoder = encoder.learn_one(x)
            xi = encoder.transform_one(x)

            # Combines the normal and encoded features.
            x = self.drop_features(x, cat_variables)
            x = x | xi

        elif (encoding == 'label'):
            x, encoder = self.__label_encoder(encoder, x, cat_variables)

        elif (encoding == 'none'):
            for feat in cat_variables:
                del x[feat]
        
        return x, encoder
        
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

    def online_model(self, data_blocks, hyper_params=None, encoding='ohe', class_pred=True, 
                     feature_engineering=True, print_freq=1):
        
        cat_variables = ["prompt_description", "month", "prompt_type", "device"]
        model = self.__init_RF(hyper_params)
        encoder = self.__init_encoder(cat_variables, encoding)
        
        test_res = {"accuracy":[], "size":[], "time":[]}

        for i, block in enumerate(data_blocks):
            
            time = perf_counter()
            
            y_true = block.pop('classification')

            if(not class_pred):
                weights = block.pop('total_weight')
                block = block.drop(['user_weight', 'prompt_weight'], axis=1)

            y_pred, model, encoder = self.core_model(block, y_true, model, cat_variables, encoder, 
                                                     encoding, class_pred, feature_engineering)
            
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

    def core_model(self, data, y, model, cat_variables, encoder, encoding, class_pred, feat_eng):
        
        y_pred_list = []

        # As the model works online, the data is accessed per record.
        for xi, yi in stream.iter_pandas(data, y):

            if(feat_eng):                        
                xi, encoder = self.prepare_data(xi, encoder, encoding, cat_variables)
            else:
                xi.drop(['date_time', 'prompt_type', 'prompt_description', 'device'], axis=1)
            
            # predict outcome and update the model
            if(class_pred):
                y_pred = model.predict_one(xi)
            else:
                y_pred = model.predict_proba_one(xi)
            
            if(y_pred == None): 
                ran = random()
                y_pred is (ran > 0.5) if class_pred else {False: ran, True: 1-ran}
            
            if(y_pred == {False: 1}):
                y_pred = {False: 1, True: 0}
            elif(y_pred == {True: 1}):
                y_pred = {False: 0, True: 1}

            y_pred_list.append(y_pred[1])
            model.learn_one(xi, yi)

        return y_pred_list, model, encoder




