import optuna
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection


from models.batch import Batch
from dataset.dataset import Dataset
from optuna.visualization.matplotlib import plot_param_importances

def opt_hyper_params():

    env = Dataset(x_users=100)
    data = env.generate_dataset(period=365, min_per_new_prompt=10)

    batch = Batch()
    y = data.pop("classification")
    X = batch.prepare_data([data], 'label')
    
    print("Finished preparing data")


    # Step 4: Running it
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, y, X), n_trials=100)
    plot_param_importances(study)


def objective(trial, y, X):
    
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 800)
    rf_max_depth = trial.suggest_int("rf_max_depth", 5, 30, log=True)
    min_samp_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_int("max_features", 10, min(len(X.columns), 50))

    classifier_obj = ensemble.RandomForestClassifier(
        max_depth=rf_max_depth, n_estimators=rf_n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features, min_samples_split=min_samp_split
        )

    # Step 3: Scoring method:
    score = model_selection.cross_val_score(classifier_obj, X, y, n_jobs=2, cv=3)
    accuracy = score.mean()
    return accuracy


