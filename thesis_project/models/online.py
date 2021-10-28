from datetime import datetime
import pandas as pd

from river import ensemble, stream, metrics, preprocessing as pp, compose
import river

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



    def model(self, data, encoding='ohe', hyper_params=None, print_every='100'):

        cat_variables = ["prompt_description", "month", "prompt_type", "device"]

        model = self.__init_RF(hyper_params)

        encoder = self.__init_encoder(cat_variables, encoding)
        metric = metrics.Precision()

        y = data.pop('classification')
        i = 0

        y_pred = None
        prev_date = None

        # As the model works online, the data is accessed per record.
        for xi, yi in stream.iter_pandas(data, y):
                                    
            show, prev_date = self.print_freq(print_every, xi, i, prev_date)

            xi, encoder = self.prepare_data(xi, encoder, encoding, cat_variables)

            # predict outcome and update the model
            y_pred = model.predict_one(xi)
            model.learn_one(xi, yi)

            # Predicting requires the model to have learned >= one.
            if (y_pred != None): 
                metric = metric.update(yi, y_pred)

            
            if (show):
                print(prev_date)
                print("#", i, round(float(metric.get() * 100), 2))
            i += 1

        return float(metric.get())


    def print_freq(self, print_every, xi, i, prev_dt):
        
        show = False
        if(isinstance(print_every, int)):
            if(i % print_every == 0):
                show = True
        else:
            date = xi['date_time'].date()
            week = xi['date_time'].isocalendar()
            month = xi['date_time'].month
            year = xi['date_time'].year

            if(prev_dt == None): 
                pass
                # return show, {'date':date, 'week':week, 'month':month, 'year':year}
            elif(print_every == 'daily'):
                if date > prev_dt['date']:
                    show = True
            elif(print_every == 'weekly'):
                if (week > prev_dt['date'] or year > prev_dt['year']):
                    show = True
            elif(print_every == 'monthly'):
                if (week > prev_dt['date'] or year > prev_dt['year']):
                    show = True
            
        return show, {'date':date, 'week':week, 'month':month, 'year':year}







    def advanced(self, X, encoding='ohe', hyper_params=None):
        
        cat_variables = ["prompt_description", "month", "prompt_type", "device"]

        model = self.__init_RF(hyper_params)

        encoder = self.__init_encoder(cat_variables, encoding)
        metric = metrics.Precision()

        y = X.pop('classification')
        i = 0

        y_pred = None

        # As the model works online, the data is accessed per record.
        for xi, yi in stream.iter_pandas(X, y):
            
            xi, encoder = self.prepare_data(xi, encoder, encoding, cat_variables)

            # predict outcome and update the model
            y_pred = model.predict_one(xi)
            model.learn_one(xi, yi)

            # Predicting requires the model to have learned >= one.
            if (y_pred != None): 
                metric = metric.update(yi, y_pred)

            if (i % 100 == 0):
                print("#", i, round(float(metric.get() * 100), 2))
            i += 1

        return float(metric.get())


