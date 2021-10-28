import numpy as np
import pandas as pd
from models.batch import Batch
from models.online import Online
from dataset.dataset import Dataset
from sklearn import metrics



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




# TODO: make the ohe working by setting a list of categories instead of guessing from the data
def system(chunk_size="daily", class_pred=False):
    
    env = Dataset(x_users=100)

    data = env.generate_dataset(period=365, exclude_weights=class_pred)


    if(isinstance(chunk_size, int)):
        data = np.array_split(data, chunk_size)
    else:
        data = split_data_on_date(data, chunk_size)

    train_x = pd.DataFrame()
    train_y = pd.Series()
    ba = Batch()

    tot_acc = 0
    tot_len = 0

    for i, chunk in enumerate(data):

        test_y = chunk.pop("classification")

        if(not class_pred):
            weights = chunk.pop('total_weight')
            chunk = chunk.drop(['user_weight', 'prompt_weight'], axis=1)

        pred_y = ba.batch_model(train_x.copy(), train_y.copy(), chunk.copy(), probability=True)

        
        train_x = train_x.append(chunk)
        train_y = train_y.append(test_y)

        if(pred_y is None):
            continue
        

        if class_pred:
            accuracy = metrics.accuracy_score(test_y, pred_y) * 100
        else:
            pred_y = [item[1] for item in pred_y]
            # print(*zip(weights, pred_y))

            accuracy = metrics.mean_absolute_error(weights, pred_y)
            # return

        tot_acc += accuracy * len(test_y)
        tot_len += len(test_y)

        # print("#", i, "accuracy:", accuracy)

        if class_pred:
            print("Block", i, "size", len(test_y), "accuracy:", round(accuracy, 3), "tot", round(tot_acc/ tot_len, 3))
        else:
            print("Block", i, "size", len(test_y), "Error:", round(accuracy, 3), "Total Average Error", round(tot_acc/ tot_len, 3))
        # print(*zip(pred_y, test_y))

    print(tot_acc/ tot_len)




system()