import random
import numpy as np
import pandas as pd
from models.batch import Batch
from models.online import Online
from dataset.dataset import Dataset
from models.online import Online
import matplotlib.pyplot as plt
import string
import copy
from itertools import zip_longest
import pickle
import easygui as g


def create_system(model_type, class_pred=True, encoding='ohe',
                  print_freq=1, feature_engineering=True, hyper_params=None, 
                  df_size=365):

    ba = Batch(encoding, hyper_params, feature_engineering)

    env = Dataset()
    data = env.generate_dataset()

    accuracy = ba.test_model(data)

    print("accuracy of the system:", accuracy)

    ba.save_model('first_model_of_the_day')


def predict_weights(model_type):

    env = Dataset()
    data = env.generate_dataset()
    data = data.sample(n=5)

    Y = data.pop('classification')

    filename = g.fileopenbox()
    file = open(filename,'rb')

    model = pickle.load(file)

    if(model_type == 'batch'):
        pred_y = model.predict(data, class_pred=False)
    
    print([pred[1] for pred in pred_y])
    # print(pred_y)



create_system('batch', encoding='ohe')
# predict_weights('batch')






