


from numpy import true_divide
from dataset.dataset import Dataset
from models.online import Online
from models.batch import Batch
import pickle
from preprocessing.batch import MultiLabelEncoder, MultiOneHotEncoder, NoEncoder

from time import perf_counter
import pandas as pd
import easygui as g

from parameter_tuning.hyperParameter import opt_hyper_params


# env = Dataset(x_users=100)
# data = env.generate_dataset(period=365, exclude_weights=True, min_per_new_prompt=10)

# ba = Batch('ohe', None, True)

# # data = ba.test_model(data)

# # print(data.columns)
# accuracy = ba.test_model(data)

# print(accuracy)

env = Dataset(x_users=100)
data = env.generate_dataset(period=365, min_per_new_prompt=10)
# Y = data.pop('classification')

# print("--------------------------------------------------------")
# print("--------------------------------------------------------")
# print("------------- OHE --------------------------------------")

# opt_hyper_params('ohe', True, data.copy())
# print("--------------------------------------------------------")
# print("--------------------------------------------------------")
# print("------------- LABEL ------------------------------------")
# opt_hyper_params('label', True, data.copy())
# print("--------------------------------------------------------")
# print("--------------------------------------------------------")
# print("------------- NONE -------------------------------------")
# opt_hyper_params('none', True, data.copy())
# print("--------------------------------------------------------")
# print("--------------------------------------------------------")
# print("------------- NO FEATURE ENGINEERING -------------------")
# opt_hyper_params('none', False, data.copy())


print("------------- OHE --------------------------------------")

ba = Batch('ohe')
accuracy = ba.test_model(data.copy())
print(accuracy)
print(ba.feature_imp)


print("------------- LABEL ------------------------------------")
ba = Batch('label')
accuracy = ba.test_model(data.copy())
print(accuracy)
print(ba.feature_imp)


print("------------- NONE -------------------------------------")

ba = Batch('none')
accuracy = ba.test_model(data.copy())
print(accuracy)
print(ba.feature_imp)


print("------------- NO FEATURE ENGINEERING -------------------")
ba = Batch('none', feature_engineering=False)
accuracy = ba.test_model(data.copy())
print(accuracy)
print(ba.feature_imp)



# 0.6217757936507936 and parameters: 
# {'rf_n_estimators': 300, 'rf_max_depth': 30, 'min_samples_split': 10, 'min_samples_leaf': 6}


# 0.6156106962177788 and parameters: 
# {'rf_n_estimators': 300, 'rf_max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 6}

# # on = Online('ohe', True, None)
# ba.save_model('banana')

# filename = g.fileopenbox()
# file = open(filename,'rb')

# on = pickle.load(file)

# print(on.feature_engineering)






# accuracy = ba.test_model(data)

# print(accuracy)



# data = env.generate_dataset(period=100, exclude_weights=True, min_per_new_prompt=10)

# accuracy = ba.test_prediction(data)
# # accuracy = ba.test_model(data)

# print(accuracy)


# print('first data before')
# print(data.head())

# data = NE.fit_transform(data)

# print('first data after')
# print(data.head())






