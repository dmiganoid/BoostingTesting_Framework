import time
import numpy as np
import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

import sys
import os

# Ща захардкодил, завтра исправлю. Тут беда с импортом из родительской директории
from os import getpid
from psutil import Process
def get_memory_usage_mb():
    process = Process(getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)


def load_algorithm(algorithm, algorithm_config, base_estimator_cfg):
    # base estimator initialization
    match base_estimator_cfg['estimator_type']:
        case "stump":
            from sklearn.tree import DecisionTreeClassifier
            base_estimator = DecisionTreeClassifier(**base_estimator_cfg['estimator_params'])
        case "neural_network":
            raise NotImplementedError

    param_grid = dict()
    algorithm_class = None
    match algorithm:
        case "AdaBoost":
            from sklearn.ensemble import AdaBoostClassifier
            algorithm_class = AdaBoostClassifier
            param_grid["estimator"] = [base_estimator]
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["algorithm"] = ['SAMME']

        case "GradientBoost":
            from sklearn.ensemble import GradientBoostingClassifier
            algorithm_class = GradientBoostingClassifier
            param_grid["loss"] = ['exponential']
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']

        case "BrownBoost":
            from models.brownboost import BrownBoost
            algorithm_class = BrownBoost
            param_grid["estimator"] = [base_estimator]
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["c"] = algorithm_config['BrownBoost']['c']
            param_grid["convergence_criterion"] = algorithm_config['BrownBoost']['convergence_criterion']
            param_grid["learning_rate"] = [1]

        case "MadaBoost":
            from models.madaboost import MadaBoost
            algorithm_class = MadaBoost
            param_grid["estimator"] = [base_estimator]
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = [1]

        # No estimator for CatBoost
        case "CatBoost":
            from catboost import CatBoostClassifier
            algorithm_class = CatBoostClassifier
            param_grid["iterations"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["depth"] = algorithm_config['CatBoost']['depth']
            param_grid["verbose"] = [False]

        # No estimator for XGBoost
        case "XGBoost":
            from xgboost import XGBClassifier
            algorithm_class = XGBClassifier
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["max_depth"] = algorithm_config['XGBoost']['max_depth']
            param_grid["use_label_encoder"] = [False]
            param_grid["eval_metric"] = ["logloss"]

        # No estimator for LightGBM
        case "LightGBM":
            from lightgbm import LGBMClassifier
            algorithm_class = LGBMClassifier
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["max_depth"] = algorithm_config['LightGBM']['max_depth']


        # <TODO find way to implement different base estimators for CatBoost, XGBoost, LightGBM>

        case "FilterBoost":
            raise NotImplementedError

    return (algorithm_class, param_grid)



class BoostingBenchmarkTrainer:
    def __init__(self, algorithms : list):
        self.algorithms = algorithms
        
    def fit_and_evaluate(self, X, y, random_state=42, test_size=0.15, results_path="results/", test_name="test"):
        results = {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
        for algorithm_class, algorithm_param_grid in self.algorithms:
            mem_before = get_memory_usage_mb()
            
            if algorithm_class == None:
                continue

            model = GridSearchCV(algorithm_class(), param_grid=algorithm_param_grid, n_jobs=-1)

            time_before = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - time_before

            mem_after = get_memory_usage_mb()
            mem_usage = mem_after - mem_before

            time_before = time.time()
            preds = model.best_estimator_.predict(X_test)
            inference_time = time.time() - time_before

            np.savetxt(f'{results_path}/{test_name}_{algorithm_class.__name__}.csv', preds, delimiter=",")

            results[algorithm_class.__name__] = {
                "model_params" : str(model.best_params_),
                "train_time_sec": train_time,
                "inference_time_sec": inference_time,
                "memory_usage_mb": mem_usage,
                "train_accuracy" : accuracy_score(model.predict(X_train), y_train),
                "test_accuracy" : accuracy_score(model.predict(X_test), y_test)
            }
            np.savetxt(f'{results_path}/{test_name}_{'train-dataset'}.csv', np.hstack((X_train, y_train.reshape(X_train.shape[0], 1))), delimiter=",")
            np.savetxt(f'{results_path}/{test_name}_{'test-dataset'}.csv', np.hstack((X_test, y_test.reshape(X_test.shape[0], 1))), delimiter=",")

        with open(f'{results_path}/{test_name}_results.json', 'w') as file:
            json.dump(results, file)
        print(f'Finished {test_name}')
        return 0
