from dataset.dataset import Dataset
import os
test_results = {10:[], 50:[], 100:[], 500:[],1000:[]}
print("START TEST")
print(os.getcwd())
x = 10
tot = 0
for key in test_results:

    print("Running for:", key, "users")
    env = Dataset(x_users=key)

    for i in range(x):

        accuracy = env.test_dataset_accuracy()
        print(accuracy)
        
        test_results[key].append(accuracy) 


        print("run:", i, "weight:", accuracy)

print("-----------------------")

for key in test_results:
    average = round(sum(test_results[key]) / x * 100, 3)
    minimum = round(min(test_results[key]) * 100, 3)
    maximum = round(max(test_results[key]) * 100, 3)

    print("users:", key, "avg:", average, "min:", minimum, "max:", maximum)

