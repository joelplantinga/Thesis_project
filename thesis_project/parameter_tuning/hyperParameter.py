import optuna
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection


from models.batch import Batch
from dataset.dataset import Dataset
from optuna.visualization.matplotlib import plot_param_importances

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from time import gmtime, strftime




def opt_hyper_params(encoding, fe, data):


    tscv = TimeSeriesSplit(n_splits=5)

    batch = Batch(encoding, hyper_params=None, feature_engineering=fe)
    y = data.pop("classification")
    X = batch.prepare_data(data)
    
    print("Finished preparing data")


    # Step 4: Running it
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, y, X, tscv), n_trials=10)

    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    time = strftime("%Y.%m.%d-%H", gmtime())

    df.to_csv(encoding+"_fe-"+str(fe)+"_"+time+"H_optuna-studyXX.csv")


    # plot_param_importances(study)

def objective(trial, y, X, tscv):

    
    rf_n_estimators = trial.suggest_categorical('rf_n_estimators', [50, 75, 100, 125, 150, 200, 250, 300, 400, 500])
    # rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 800)
    
    # rf_max_depth = trial.suggest_int("rf_max_depth", 5, 30, log=True)
    rf_max_depth = trial.suggest_categorical('rf_max_depth', [2, 5, 10, 20, 30, 40 ,50])

    # min_samp_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samp_split = trial.suggest_categorical('min_samples_split', [2, 5, 10])

    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 6)

    # max_features = trial.suggest_int("max_features", 5, min(len(X.columns), 50))
    # max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt'])



    classifier_obj = ensemble.RandomForestClassifier(
        max_depth=rf_max_depth, n_estimators=rf_n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features='auto', min_samples_split=min_samp_split
        )

    # Step 3: Scoring method:
    score = cross_val_score(classifier_obj, X, y, n_jobs=2, cv=tscv)
    accuracy = score.mean()
    return accuracy


