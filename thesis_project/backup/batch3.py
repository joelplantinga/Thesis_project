from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import FeatureHasher

from preprocessing.batch import MultiLabelEncoder, MultiOneHotEncoder, NoEncoder

class Batch:


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

    def __vis_feat(self, feature_imp):
        
        # Creating a bar plot
        sns.barplot(x=feature_imp, y=feature_imp.index)
        # Add labels to your graph
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend()
        plt.show()

    def __label_encoder(self, df, features):

        """Function that uses the label encoding technique for the 
        categorical features.

        Attributes:
        ------------

        df : pd.Dataframe
            Dataset that containing categorical features.

        features : list[str]
            List of feature names

        Returns:
        ------------

        df : pd.Dataframe
            Dataset containing label encoded features 

        """        
        
        LE = LabelEncoder()

        for feature in features:

            df[feature] = LE.fit_transform(df[feature])

        return df

    def __feature_hasher(self, df, features):
        
        """Function that uses the hashing technique for the 
        categorical features.

        Attributes:
        ------------

        df : pd.Dataframe
            Dataset that containing categorical features.

        features : list[str]
            List of feature names.

        Returns:
        ------------

        df : pd.Dataframe
            Dataset containing hashed features.

        """        
        
        for feature, n_feat in features:
            
            FH = FeatureHasher(n_features=n_feat, input_type='string')

            df[feature] = df[feature].apply(str)
        
            hashed_features = FH.fit_transform(df[feature])

            hashed_features = pd.DataFrame(hashed_features.toarray())
            hashed_features = hashed_features.add_prefix(feature + "_")

            df = pd.concat([df.reset_index(drop=True), hashed_features], axis=1)
        
        df.to_csv("output/hashing.csv")

        return df

    def __init_RF(self, params, x_cols):
        
        if params is None:
            forest = RandomForestClassifier(n_estimators=78, 
                                            max_depth=8, 
                                            min_samples_split=8,
                                            min_samples_leaf=4, 
                                            max_features=min(17, x_cols))
        elif(params == "basic"):
            forest = RandomForestClassifier(n_estimators=100)
        elif(params == "label"):
            forest = RandomForestClassifier(n_estimators=255, 
                                max_depth=5, 
                                min_samples_split=2,
                                min_samples_leaf=1, 
                                max_features=min(11, x_cols))
        else:
            forest = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                            max_depth=params["max_depth"], 
                                            min_samples_split=params["min_samples_split"],
                                            min_samples_leaf=params["min_samples_leaf"], 
                                            max_features=min(params["max_features"], x_cols))
        return forest

    def __ohe(self, data, OHE, i, cols):
        
        #TODO: If time allows; include feature names correctly (X0_question -> prompttype_question) see doc for inu
        dummy_data = np.array(data[cols])
        if(i == 0):
            dummy_data = OHE.fit_transform(dummy_data).toarray()
        else:
            dummy_data = OHE.transform(dummy_data).toarray()
            
        dummy_data = pd.DataFrame(dummy_data, columns=OHE.get_feature_names())
        data = data.drop(cols, axis=1)
        data = data.reset_index()
        data = pd.concat([data, dummy_data], axis=1)

        return data

    def prepare_data(self, datasets, encoding, feature_engineering):

        """Function that prepares the data before it gets passed to 
        the Random Forest model. It calculates the distance and floor
        difference between user and prompt. Modifies the date feature and
        encodes the categorical data in the encoder given by the user using 
        encoding='ENCODER'.

        Attributes:
        ------------

        x_data : pd.Dataframe
            Dataset of all the features.

        encoding : str
            Encoding technique for the categorical variables. Either 'label' for 
            label encoding, 'ohe' for One Hot Encoding, 'hashing' for hashing, 'combi' 
            for a combination of both hashing and OHE and none for removing the categorical
            variables.

        Returns:
        ------------

        x_data : pd.Dataframe
            Dataset that is rightly configured for the model.
        
        """

        # TODO: user_id is removed from no_encoding. -> BUG
        cat_variables = ['prompt_type', 'device', 'user_id', 'prompt_description']
        hash_features = [('user_id', 8), ("prompt_description", 3), 
                        ("prompt_type", 1), ("device", 3)]
        
        
        OHE = OneHotEncoder(handle_unknown='ignore')
        out = []
        for i, x_data in enumerate(datasets):

            # if(x_data.empty):
            #     print("dataframe is empty")
            #     continue

            if(feature_engineering):

                x_data = self.__calc_dist(x_data.copy())        
                x_data = self.__calc_floor_dif(x_data.copy())   
                x_data = self.__date_mod(x_data.copy()) 

                if (encoding == 'label'):
                    x_data = self.__label_encoder(x_data, cat_variables)
                
                elif (encoding == 'ohe'):
                    x_data = self.__ohe(x_data, OHE, i, cat_variables)

                elif (encoding == 'hashing'):
                    x_data = self.__feature_hasher(x_data, hash_features)
                    x_data = x_data.drop(cat_variables, axis=1)
                
                elif (encoding == 'combi'):
                    x_data = self.__ohe(x_data, OHE, i, ['prompt_type', 'device', 'prompt_description'])
                    x_data = self.__feature_hasher(x_data, [('user_id', 10)])
                    x_data = x_data.drop(['user_id'], axis=1)
                
                else:
                    x_data = x_data.drop(['prompt_type', 'device', 'prompt_description'], axis=1)
            else:
                x_data = x_data.drop(['date_time', 'prompt_type', 'prompt_description', 'device'], axis=1)

            
            out.append(x_data)
        
        if(len(out) == 1):
            return (out[0])
        else:
            return (out[0], out[1])

    def init_encoding(self, encoding):

            if(encoding == 'ohe'):
                encoder = MultiOneHotEncoder(['prompt_type', 'device', 'user_id', 'prompt_description'])
            elif(encoding == 'label'):
                encoder = MultiLabelEncoder(['prompt_type', 'device', 'prompt_description'])
            elif(encoding == 'none'):
                encoder = NoEncoder(['prompt_type', 'device', 'prompt_description'])

            return encoder

    def prepare_dataNEW(self, x_data, encoder, feature_engineering):

        # TODO: user_id is removed from no_encoding. -> BUG
        # cat_variables = ['prompt_type', 'device', 'user_id', 'prompt_description']
        # hash_features = [('user_id', 8), ("prompt_description", 3), 
        #                 ("prompt_type", 1), ("device", 3)]
        
        if(feature_engineering):

            x_data = self.__calc_dist(x_data.copy())        
            x_data = self.__calc_floor_dif(x_data.copy())   
            x_data = self.__date_mod(x_data.copy()) 

            x_data = encoder.fit_transform(x_data)
                        
        else:
            x_data = x_data.drop(['date_time', 'prompt_type', 'prompt_description', 'device'], axis=1)

        return x_data, encoder

            


    def test_model(self, data, hyper_params=None, feature_engineering=True, encoding='ohe', test_size=0.5):

        y = data.pop('classification')

        train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=test_size)

        pred_y = self.core_model(train_x, train_y, test_x, hyper_params=hyper_params,
                                  feature_engineering=feature_engineering, encoding=encoding)
        
        accuracy = metrics.accuracy_score(test_y, pred_y) * 100

        return accuracy

    def batch_system(self, data, class_pred, encoding='ohe', hyper_params=None, feature_engineering=True, print_freq=1):
        
        train_x = pd.DataFrame()
        train_y = pd.Series()

        test_res = {"accuracy":[], "size":[], "time":[]}

        for i, chunk in enumerate(data):
            
            time = perf_counter()

            test_y = chunk.pop("classification")

            if(not class_pred):
                weights = chunk.pop('total_weight')
                chunk = chunk.drop(['user_weight', 'prompt_weight'], axis=1)

            pred_y = self.core_model2(train_x = train_x.copy(), train_y=train_y.copy(), test_x=chunk.copy(), 
                                      hyper_params=hyper_params, feature_engineering=feature_engineering, 
                                      encoding=encoding, class_pred=class_pred)

            train_x = train_x.append(chunk)
            train_y = train_y.append(test_y)

            time_measured = perf_counter() - time

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
                      "time:", round(time_measured, 3)) # "tot acc", round(tot_acc / tot_len, 3)

        return test_res



    def core_model(self, train_x, train_y, test_x, hyper_params=None, 
                    feature_engineering=True, encoding='ohe', class_pred=True):
                
        if(train_x.empty or train_y.empty):
            return None
            
        train_x, test_x = self.prepare_data([train_x, test_x], encoding, feature_engineering) 

        RF = self.__init_RF(hyper_params, len(train_x.columns))
        
        RF.fit(train_x,train_y)

        # feature_imp = pd.Series(RF.feature_importances_,index=train_x.columns).sort_values(ascending=False)
        # print(feature_imp)

        if(class_pred):
            pred_y = RF.predict(test_x)
        else:    
            pred_y = RF.predict_proba(test_x)

        return pred_y
        


    def core_model2(self, activity='both', train_x=None, train_y=None, 
                    test_x=None, hyper_params=None, feature_engineering=True, 
                    class_pred=True, RF=None, encoder = None):
                
        if(activity == 'train' or activity == 'both'):
            
            train_x, encoder = self.prepare_dataNEW(train_x, encoder, feature_engineering)
            
            if(train_x.empty or train_y.empty):
                print("RETURN: Cannot train model on empty dataset.")
                return None

            RF = self.__init_RF(hyper_params, len(train_x.columns))
            RF.fit(train_x,train_y)

            if(activity == 'train'):
                return RF, None

        if(activity == 'predict' or activity == 'both'):

            test_x, encoder = self.prepare_dataNEW(test_x, encoder, feature_engineering)

            if(class_pred):
                pred_y = RF.predict(test_x)
            else:    
                pred_y = RF.predict_proba(test_x)

            return RF, pred_y







        if(activity == 'train'):
            return RF
        







    




        # hyper_params = {"n_estimators":78, "max_depth":8, "min_samples_split":8, 
        #                 "min_samples_leaf":4, "max_features":min(17, max_cols)}
