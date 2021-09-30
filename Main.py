from Online import Online
from Dataset import Dataset
from time import perf_counter




print("-----------------------")
print("MAKE DATASET")

env = Dataset(x_users=100)
data = env.generate_prompts(period=365, min_per_new_prompt=10)
data = env.finish_dataset(data)


print("START TEST")

on = Online()

for i in range(2):

    time = perf_counter()

    if(i == 0):
        
        acc = on.basic(data.copy())
        print("#BASIS Total accuracy: ", round(acc * 100, 2), "%%", sep='')

    else:
        acc = on.first_modification(data.copy())
        print("#ENHANCED Total accuracy: ", round(acc * 100, 2), "%%", sep='')


        print("Time used: ", round(perf_counter() - time, 2), sep='')
        print("-----------------------")


