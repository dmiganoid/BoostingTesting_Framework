import time
import numpy as np
import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, ParameterGrid
from multiprocessing import Pool, cpu_count
import sys
from os import mkdir
from functools import wraps
# Ща захардкодил, завтра исправлю. Тут беда с импортом из родительской директории
from os import getpid
from psutil import Process


def get_memory_usage_mb():
    process = Process(getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)


def timed_wrapper(func, *args, **kwargs): # picklable workaround
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time-start_time

def train_test_model(algorithm_class, params, X_train, X_test, y_train, y_test, results_path):
    model = algorithm_class(**params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time


    start_time = time.time()
    model.predict(X_test)
    inference_time = time.time() - start_time

    np.savetxt(f'{results_path}/train_{algorithm_class.__name__}.csv', model.predict(X_train), delimiter=",")
    np.savetxt(f'{results_path}/test_{algorithm_class.__name__}.csv', model.predict(X_test), delimiter=",")
    np.savetxt(f'{results_path}/train-dataset.csv', np.hstack((X_train, y_train.reshape(X_train.shape[0], 1))), delimiter=",")
    np.savetxt(f'{results_path}/test-dataset.csv', np.hstack((X_test, y_test.reshape(X_test.shape[0], 1))), delimiter=",")
 
    output_params = model.get_params()
    if 'estimator' in output_params.keys():
        output_params['estimator'] = str(output_params['estimator'])
    results = {
                    "model_params" : output_params,
                    "train_time_sec": train_time,
                    "inference_time_sec": inference_time,
                    "memory_usage_mb": sys.getsizeof(model),
                    "train_accuracy" : accuracy_score(model.predict(X_train), y_train),
                    "test_accuracy" : accuracy_score(model.predict(X_test), y_test)
                }
    return results

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

        # No estimator for GradientBoosting
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
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']

        case "MadaBoost":
            from models.madaboost import MadaBoost
            algorithm_class = MadaBoost
            param_grid["estimator"] = [base_estimator]
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']

        # No estimator for CatBoost
        case "CatBoost":
            from catboost import CatBoostClassifier
            algorithm_class = CatBoostClassifier
            param_grid['silent'] = [True]
            param_grid["iterations"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["depth"] = algorithm_config['CatBoost']['depth']

        # No estimator for XGBoost
        case "XGBoost":
            from xgboost import XGBClassifier
            algorithm_class = XGBClassifier
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["max_depth"] = algorithm_config['XGBoost']['max_depth']
            param_grid["use_label_encoder"] = [False]
            param_grid["eval_metric"] = ["logloss"]
            param_grid['verbosity'] = [0]

        # No estimator for LightGBM
        case "LightGBM":
            from lightgbm import LGBMClassifier
            algorithm_class = LGBMClassifier
            param_grid['verbose'] = [-1]
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
        
    def fit_and_evaluate(self, X, y, random_state=42, test_size=0.15, results_path="results", test_name="test"):
        results = {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
        pool = Pool(processes=cpu_count()-1)
        print(f"Starting {test_name}")
        mkdir(f'{results_path}/{test_name}')
        for algorithm_class, algorithm_param_grid in self.algorithms:
            if algorithm_class == None:
                continue
            threads = []
            print(f"Training {algorithm_class.__name__}")
            for params in ParameterGrid(algorithm_param_grid):
                threads.append(pool.apply_async(train_test_model, args=[algorithm_class, params, X_train, X_test, y_train, y_test, f'{results_path}/{test_name}']))
            for model_i in range(len(threads)):
                results[f"{algorithm_class.__name__}_{model_i}"] = threads[model_i].get(timeout=None)
        pool.close()
        with open(f'{results_path}/{test_name}/results.json', 'w') as file:
            json.dump(results, file)
        print(f'Finished {test_name}')
        return 0
