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


def split_data_on_date(data, frequency="daily"):

    if(frequency == "daily"):

        cols = ["date"]
        data['date'] = data['date_time'].dt.date

        data = data.groupby(['date'])

    elif(frequency == "weekly"):

        cols = ["week", "year"]
        data['week'] = data['date_time'].dt.isocalendar().week
        data['year'] = data['date_time'].dt.year

        data = data.groupby(['week', 'year'])

    elif(frequency == "monthly"):

        cols = ["month", "year"]
        data['month'] = data['date_time'].dt.month
        data['year'] = data['date_time'].dt.year

        data = data.groupby(['month', 'year'])

    data = [data.get_group(x) for x in data.groups]

    # Helper columns must be deleted again
    data = [x.drop(cols, axis=1) for x in data]

    return data

def split_data_int(data, n):

    list = []
    for i in range(0, len(data), n):
        list.append(data[i:i+n])

    return list

def make_data(chunk_size, class_pred, df_size):
    env = Dataset(x_users=100)
    data = env.generate_dataset(period=df_size, exclude_weights=class_pred)

    if(isinstance(chunk_size, int)):
        data = split_data_int(data, chunk_size)
    else:
        data = split_data_on_date(data, chunk_size)
    return data

def print_info(chunk_size, class_pred, model_type, encoding):
    print("-----------------------------------")
    print("START OF THE SYSTEM")
    print("Running in", model_type, "mode")

    print("Running system in chunks of size:", chunk_size)
    print("Using", encoding, "as feature encoder")

    if class_pred:
        print("Testing system using class predictions")
    else:
        print("Testing system using probability difference")
    print("-----------------------------------")

def test_encoding(model_type, chunk_size="daily", class_pred=True, print_freq=10, 
                  hyper_params=None, df_size=365, x=3):

    results = {"No_feature_engineering": [],
               "label": [],
               "ohe": [],
               "none": []
               }
    out = {}

    for i in range(x):
        print("XXXXXXX TEST ROUND", i + 1, "/", x,  "XXXXXXXX")
        data = make_data(chunk_size, class_pred, df_size)
        
        for key in results:

            if(key == "No_feature_engineering"):
                res = run_system(model_type, chunk_size, class_pred, 'none',
                                 print_freq, False, hyper_params,
                                 df_size, copy.deepcopy(data), False, True)
            else:
                res = run_system(model_type, chunk_size, class_pred, key,
                                 print_freq, True, hyper_params,
                                 df_size, copy.deepcopy(data), False, True)

            results[key].append(res)
    
    for key in results:
        df = pd.DataFrame()
        i = 0
        for res in results[key]:
            
            i += 1

            if (len(df) == 0):
                df = res
                continue

            df['index'] = df['index'] if (
                len(df['index']) > len(res['index'])) else res['index']
            
            old_size = df['size'].copy()
            
            df['size'] = add_cols(res['size'],old_size)
            
            df['accuracy'] = add_cols(
                df['accuracy'] * old_size, res['accuracy'] * res['size']) / df['size']

            df['cum_accuracy'] = add_cols(
                df['cum_accuracy'] * old_size.cumsum(), res['cum_accuracy'] * res['size'].cumsum()) / \
                    df['size'].cumsum()

            df['time'] = add_cols(df['time'] * (i-1), res['time']) / i

            df['cum_time'] = add_cols(df['cum_time'] * (i-1), res['cum_time']) / i
            
        df.to_csv("plots/encoder_test_" + model_type + "_" + key + ".csv")    
        out[key] = df

    plot_encoding_test(out)

def plot_encoding_test(results):
    
    
    colors = ['red', 'brown', 'darkred', 'firebrick', 'red', 'tomato', 'lightcoral']
    y_axis = {'name': 'Accuracy', 'data': []}

    for i, key in enumerate(results):
        df = results[key]
        y_axis['data'].append(
            {'obj': df['cum_accuracy'], 'name': key, 'col': colors[i]}
        )

        if(i==0):
            x_axis = {'obj': df['index'], 'name': 'Measurements', 'col': None}

    make_graph(x_axis, y_ax_1=y_axis)
    
    plt.show()
    
def add_cols(col1, col2):
    return np.array([sum(x) for x in zip_longest(col1, col2, fillvalue=0)]).copy()

def system(model_type, chunk_size="daily", class_pred=True, encoding='ohe',
           print_freq=1, feature_engineering=True, hyper_params=None, 
           df_size=365, plot_data=False, make_csv=True):

    data = make_data(chunk_size, class_pred, df_size)

    results = []
    if(model_type == "both"):
        results.append(run_system('online', chunk_size, class_pred, encoding, print_freq,
                                  feature_engineering, hyper_params, df_size, copy.deepcopy(data), 
                                  plot_data, make_csv))
        
        results.append(run_system('batch', chunk_size, class_pred, encoding, print_freq,
                                  feature_engineering, hyper_params, df_size, data, 
                                  plot_data, make_csv))
    else:
        results.append(run_system(model_type, chunk_size, class_pred, encoding, print_freq,
                                  feature_engineering, hyper_params, df_size, data, 
                                  plot_data, make_csv))

    if(plot_data):
        plt.show()

    return(results)


def run_system(model_type, chunk_size, class_pred, encoding, print_freq, 
               feature_engineering, hyper_params, df_size, data, plot_data, make_csv):

    print_info(chunk_size, class_pred, model_type, encoding)

    if(model_type == "batch"):
        ba = Batch(encoding, hyper_params, feature_engineering)
        results = ba.batch_system(data, class_pred, print_freq=print_freq)

    elif(model_type == "online"):
        on = Online(encoding, feature_engineering, hyper_params)
        results = on.online_model(data, class_pred=class_pred, print_freq=print_freq)

    results = pd.DataFrame.from_dict(results)

    results['cum_accuracy'] = (
        (results['accuracy'] * results['size']).cumsum()) / results['size'].cumsum()
    results['cum_time'] = results['time'].cumsum()
    results['index'] = results.index
    results['cum_time'] = results['time'].cumsum() / (results['index'] + 1)


    if(make_csv):
        ran_string = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=3))
        filename = model_type + "_enc-" + encoding + "_cs-" + \
            str(chunk_size) + "_dfs-" + str(df_size) + "_" + ran_string + ".csv"

        results.to_csv("plots/" + filename)

    if not(plot_data):
        return results

    make_graph({'obj': results['index'], 'name': "Measurements", 'col': None},
               y_ax_1={'name': 'Accuracy',
                       'data': [{'obj': results['accuracy'], 'name': "Accuracy", 'col': 'red'},
                                {'obj': results['cum_accuracy'], 'name': "Cumulative accuracy", 'col': 'darkred'}]},
               y_ax_2={'name': 'Time',
                       'data': [{'obj': results['cum_time'], 'name': "Cumulative average time ", 'col': 'blue'}]})

    return results


def make_graph(x_axis, y_ax_1, y_ax_2=None):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(x_axis['name'], fontsize=14)
    for axis in y_ax_1['data']:
        ax.plot(x_axis['obj'], axis['obj'], color=axis['col'],
                marker="o", label=axis['name'])

    ax.set_ylabel(y_ax_1['name'], color=axis['col'], fontsize=14)

    if(y_ax_2 != None):
        ax2 = ax.twinx()
        for axis in y_ax_2['data']:
            ax2.plot(x_axis['obj'], axis['obj'], color=axis['col'],
                    marker="o", label=axis['name'])

        ax2.set_ylabel(y_ax_2['name'], color=axis['col'], fontsize=14)

    fig.legend(loc='upper left')
    plt.show(block=False)
    # save the plot as a file
    # fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
    #             format='jpeg',
    #             dpi=100,
    #             bbox_inches='tight')


def test_encoding_simple(model_type, df_size=365, x=3):

    results = {"base_model": [],
               "No_feature_engineering": [],
               "label": [],
               "ohe": [],
               "none": []
               }
    out = {}

    for i in range(x):
        print("XXXXXXX| TEST ROUND", i + 1, "/", x,  "|XXXXXXXX")

        env = Dataset()
        data = env.generate_dataset(period=df_size)
        
        for key in results:

            if(key == "No_feature_engineering"):
                ba = Batch('none', feature_engineering=False, hyper_params='no_ft')
            elif(key == "base_model"):
                ba = Batch('none', feature_engineering=False)
            else:
                ba = Batch(key, feature_engineering=True, hyper_params=key)

            res = ba.test_model(data.copy())
            
            print("encoder:", key, ", accuracy:", res)
            results[key].append(res)

    for key in results:

        print(key +":", sum(results[key]) / len(results[key]))
    
    df = pd.DataFrame()

    df = df.from_dict(results)
        
    df.to_csv("plots/encoder_test_simple" + ".csv")    





test_encoding_simple('batch', x=30)


# test_encoding('batch', chunk_size='weekly', print_freq="monthly", df_size=200, x=1)
# system('batch', chunk_size="weekly", class_pred=False,
#            df_size=100, print_freq=1, encoding="none")
# new_system('online', chunk_size=40, class_pred=False, df_size=365, print_freq=5)


# TODO: currently, combi encoding is not working
# TODO: make it work for the final goal; it should be able to first train the model on some data
# ----- and then predict on some other data. Core functions should return model.
# TODO: Currently, online is working better without feature engineering. Probably due to 
#  ---- hyper_parameters that are passed when configured with None.
# TODO: Add more hyper_param configs for different encoding configs.
# NOTE: what should we do with the first step of batch. The prediction is always None as no
#  ---- modeled prediction can be made. Should we keep it like this or make predictions?
# NOTE: How is the cum_accuracy named correctly? Cumulative is not the right word imo.


# system('online', chunk_size="daily", df_size=20, class_pred=True, feature_engineering=True, encoding='ohe')
