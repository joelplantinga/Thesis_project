from dataset import Dataset
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

        data = env.generate_prompts(period=365, min_per_new_prompt=10)
        weights = data["total_weight"]
        
        avg_weight = 0

        for weight in weights:
            avg_weight += max(weight, 1 - weight)
        
        avg_weight /= len(weights)
        
        test_results[key].append(avg_weight) 


        print("run:", i, "weight:", avg_weight)

print("-----------------------")

for key in test_results:
    average = round(sum(test_results[key]) / x * 100, 3)
    minimum = round(min(test_results[key]) * 100, 3)
    maximum = round(max(test_results[key]) * 100, 3)

    print("users:", key, "avg:", average, "min:", minimum, "max:", maximum)

