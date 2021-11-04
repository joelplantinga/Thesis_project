


from dataset.dataset import Dataset
from models.online import Online
from models.batch import Batch

from time import perf_counter
import pandas as pd


from parameter_tuning.hyperParameter import opt_hyper_params


opt_hyper_params()

# print("-----------------------")

# test_results = {"basic":[], "label":[], "ohe":[], "hash":[], "combi":[], "none":[]}


# env = Dataset(x_users=100)

# print("START TEST")

# on = Online()
# batch = Batch()

# # data = env.generate_prompts(period=365, min_per_new_prompt=10)
# # data = env.finish_dataset(data)

# # print("classification == True", len(data[data.classification == True]))
# # print("total size:", len(data))


# x = 1
# tot = 0

# for i in range(x):

#     print("Round", i+1)
#     time = perf_counter()


#     data = env.generate_dataset(period=365)  

#     # on.advanced(data.copy(), encoding='label')
#     # batch.enhanced(data.copy(), encoding='label')
#     # acc = batch.test_model(data.copy(), encoding='label')
#     # print("label", acc)
#     # acc = batch.test_model(data.copy(), encoding='ohe')
#     # print("ohe", acc)
#     # acc = batch.test_model(data.copy(), encoding='combi')
#     # print("combi", acc)
#     # acc = batch.test_model(data.copy(), encoding='hashing')
#     # print("hashing", acc)

#     acc = on.model(data.copy(), encoding='label', print_every='daily')
#     print("hashing", acc)

#     continue
#     for key in test_results:

#         if key == 'basic':
#             test_results[key].append(batch.basic(data.copy()))
#         else:
#             test_results[key].append(batch.enhanced(data.copy(), encoding=key))    

#     print("Time used: ", round(perf_counter() - time, 2), sep='')
#     print("-----------------------")



# """
# print(tot/x)
# print("Average accuracies over", x, "rounds:")
# for key in test_results:
#     average = round(sum(test_results[key]) / x, 3)
#     minimum = round(min(test_results[key]), 3)

#     print(key.upper(), "| ACCURACY -> AVG:", average, "MIN:", minimum)
#     print([round(acc, 2) for acc in test_results[key]])
#     print("-----------------------")

# """





