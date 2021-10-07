from datetime import datetime
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import FeatureHasher

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
        df['dist'] = round(pow(pow(df['room_x'] - df['user_x'], 2) + 
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

        df['floor_div'] = abs(df['user_floor'] - df['room_floor'])

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

        df["diff_in_time"] = [ (dt - unix).days for dt in date ]

        df["month_diff"] = [ abs(6 - min(dt.month, 12-dt.month)) for dt in date ] 
        
        del df["date_time"]

        return df

    # def __date_mod(self, df):

    #     date = pd.to_datetime(df["date_time"], format='%Y-%m-%d %H:%M:%S')
    #     unix = datetime.strptime("1970-01-01", "%Y-%m-%d")

    #     df["diff_in_time"] = [ (dt - unix).days for dt in date ]

    #     df["month_diff"] = [ abs(6 - min(dt.month, 12-dt.month)) for dt in date ] 
        
    #     del df["date_time"]

    #     return df

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

        return df
        

    def prepare_data(self, data, encoder):

        """Function that prepares the data before it gets passed to 
        the Random Forest model. It calculates the distance and floor
        difference between user and prompt. Modifies the date feature and
        encodes the categorical data in the encoder given by the user using 
        encoder='ENCODER'.

        Attributes:
        ------------

        df : pd.Dataframe
            Dataset that containing both the features and classification.

        encoder : str
            Encoding technique for the categorical variables. Either 'label' for 
            label encoding, 'ohe' for One Hot Encoding, 'hashing' for hashing, 'combi' 
            for a combination of both hashing and OHE and none for removing the categorical
            variables.

        Returns:
        ------------

        X : pd.Dataframe
            Dataset containing the features.
        
        y : pd.Dataframe
            The results of the dataset.

        """

        # not_imp = ['has_location']
        # data = data.drop(not_imp, axis=1)

        y = data.pop('classification')

        X = data

        X = self.__calc_dist(X)
        X = self.__calc_floor_dif(X)
        X = self.__date_mod(X)

        cat_variables = ['prompt_type', 'device', 'user_id', 'prompt_description']
        hash_features = [('user_id', 8), ("prompt_description", 3), 
                         ("prompt_type", 1), ("device", 3)]

        if (encoder == 'label'):
            X = self.__label_encoder(X, cat_variables)
        elif (encoder == 'ohe'):
            X = pd.get_dummies(X, columns = cat_variables)
        elif (encoder == 'hashing'):
            X = self.__feature_hasher(X, hash_features)
            X = X.drop(cat_variables, axis=1)
        elif (encoder == 'combi'):
            X = pd.get_dummies(X, columns = ['prompt_type', 'device', 'prompt_description'])
            X = self.__feature_hasher(X, [('user_id', 10)])
            X = X.drop(['user_id'], axis=1)
        else:
            X = X.drop(cat_variables, axis=1)

        return X, y

    def enhanced(self, data, encoder='combination'):

        """The enhanced Random Forest model. Given the data and the 
        encode technique, it prepares the data and makes a model that 
        tries to predict whether a prompt will be performed. The result 
        is an accuracy of the model.

        Attributes:
        ------------

        data : pd.Dataframe
            Dataset contains both features and classification.

        encoder : str
            encoding technique

        Returns:
        ------------

        accuracy : float
            Percentage that indicates the accuracy of the model.
        """       

        X, y = self.prepare_data(data, encoder)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        forest = RandomForestClassifier(n_estimators=130, max_depth=5, min_samples_split=9,
        min_samples_leaf=10, max_features=min(11, len(X.columns)))

        forest.fit(X_train,y_train)

        feature_imp = pd.Series(forest.feature_importances_,index=X.columns).sort_values(ascending=False)

        # print(feature_imp)
        # self.__vis_feat(feature_imp)

        y_pred = forest.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred) * 100
        print("BATCH ENHANCED", encoder, "Accuracy:", round(accuracy, 3))

        return accuracy

    def basic(self, data):

        """The basic Random Forest model. Given the data and the 
        encode technique, it removes the categorical variables and makes a 
        model that tries to predict whether a prompt will be performed. 
        The result is an accuracy of the model.

        Attributes:
        ------------

        data : pd.Dataframe
            Dataset contains both features and classification.

        Returns:
        ------------

        accuracy : float
            Percentage that indicates the accuracy of the model.
        """        

        data = data.drop(['date_time', 'prompt_type', 'prompt_description', 'device'], axis=1)

        y = data.pop('classification')
        X = data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        forest = RandomForestClassifier(n_estimators=100)

        forest.fit(X_train,y_train)

        y_pred = forest.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred) * 100
        print("BATCH BASIC Accuracy:", round(accuracy, 3))

        return accuracy

