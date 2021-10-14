import optuna
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
from Batch import Batch
from Dataset import Dataset
from optuna.visualization.matplotlib import plot_param_importances
from Online import Online

env = Dataset(x_users=100)
data = env.generate_prompts(period=365, min_per_new_prompt=10)
data = env.finish_dataset(data)

on = Online()

print("Finished preparing data")


#Step 1. Define an objective function to be maximized.
def objective(trial):
    
    hyper_params = {
        "n_models": trial.suggest_int("n_models", 20, 100),
        "max_features": trial.suggest_int("max_features", 8, 12),
        "max_depth": trial.suggest_int("max_depth", 5, 30, log=True),
    }

    accuracy = on.advanced(data.copy(),'label', hyper_params)

    return accuracy

# Step 4: Running it
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=70)
plot_param_importances(study)

