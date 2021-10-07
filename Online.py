from datetime import datetime
import pandas as pd

from river import ensemble, stream, metrics, preprocessing as pp, compose

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

    def first_modification(self, X):
        
        model = ensemble.AdaptiveRandomForestClassifier( 
            n_models=3,
            seed=42
        )

        # For testing purposes
        del X['prompt_type']
        del X['device']


        OHE_features = ["prompt_description", "month"]

        # OneHotEncoding pipe. First the right features are chosen
        # before they are passed to the encoder.
        OHE = compose.Select( *OHE_features) | pp.OneHotEncoder()

        # The metric measures how good the models performs.
        metric = metrics.Precision()

        y = X.pop('classification')
        i = 0

        # As the model works online, the data is accessed per record.
        for xi, yi in stream.iter_pandas(X, y):
            
            # Make sure that the features are understood well by the model.
            xi = self.__calc_dist(xi)     
            xi = self.__calc_floor_dif(xi)
            xi = self.__date_mod(xi)

            # adds the features to the encoder. In order to memorise
            # what categories have passed (in online only one category 
            # at the time is seen).
            OHE = OHE.learn_one(xi)

            # The category is encoded in binary features.
            xii = OHE.transform_one(xi)

            # Combines the normal and encoded features.
            xi = self.drop_features(xi, OHE_features)
            xi = xi | xii

            # predict outcome and update the model
            y_pred = model.predict_one(xi)
            model.learn_one(xi, yi)

            # The first time, the model cannot predict and therefore returns
            # None. None however, is not accepted by update().
            if(y_pred != None):
                metric = metric.update(yi, y_pred)

            # For testing purposes
            if (i % 5000 == 0):
                print("#", i, round(float(metric.get() * 100), 2))
            i += 1
        
        return float(metric.get())




