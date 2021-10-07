from Online import Online
from Dataset import Dataset
from time import perf_counter
from Batch import Batch
import pandas as pd


print("-----------------------")

test_results = {"basic":[], "label":[], "ohe":[], "hash":[], "combi":[], "none":[]}


env = Dataset(x_users=100)


print("START TEST")

on = Online()
batch = Batch()

x = 10

for i in range(x):

    print("Round", i+1)
    time = perf_counter()

    data = env.generate_prompts(period=365, min_per_new_prompt=10)
    data = env.finish_dataset(data)

    for key in test_results:

        if key == 'basic':
            test_results[key].append(batch.basic(data.copy()))
        else:
            test_results[key].append(batch.enhanced(data.copy(), encoder=key))    

    print("Time used: ", round(perf_counter() - time, 2), sep='')
    print("-----------------------")


print("Average accuracies over", x, "rounds:")
for key in test_results:
    average = round(sum(test_results[key]) / x, 3)
    minimum = round(min(test_results[key]), 3)

    print(key.upper(), "| AVG ACC :", average, "MIN:", minimum)
    print([round(acc, 2) for acc in test_results[key]])
    print("-----------------------")


