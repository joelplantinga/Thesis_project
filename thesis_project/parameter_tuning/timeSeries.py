from sklearn import ensemble
from time import perf_counter
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import optuna

from models.batch import Batch
from dataset.dataset import Dataset


env = Dataset(x_users=100)
data = env.generate_prompts(period=365, min_per_new_prompt=10)
data = env.finish_dataset(data)

ba = Batch()

X, y = ba.prepare_data(data, encoding='combi')

tscv = TimeSeriesSplit(n_splits=5)

#Step 1. Define an objective function to be maximized.
def objective(trial):
    
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 700, 710)
    rf_max_depth = trial.suggest_int("rf_max_depth", 5, 30, log=True)
    min_samp_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
    max_features = trial.suggest_int("max_features", 10, min(50, len(X.columns)))

    
    classifier_obj = ensemble.RandomForestClassifier(
        max_depth=rf_max_depth, n_estimators=rf_n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features, min_samples_split=min_samp_split
        )

    # Step 3: Scoring method:
    score = cross_val_score(classifier_obj, X, y, n_jobs=2, cv=tscv)
    accuracy = score.mean()
    return accuracy

# Step 4: Running it
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150)
# plot_param_importances(study)
