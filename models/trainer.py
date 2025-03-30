import numpy as np
import pandas as pd
import json
import os
import time
from psutil import cpu_count
from sklearn.model_selection import train_test_split, ParameterGrid
from multiprocessing import Pool
from sklearn.metrics import accuracy_score

def train_test_model(algorithm_class, params, X_train, X_test, y_train, y_test, results_path, ind="", random_state=None, save_predictions=True):
    model = algorithm_class(**params)
    st = time.time()
    model.fit(X_train, y_train)
    tr_time = time.time() - st
    st = time.time()
    model.predict(X_test)
    inf_time = time.time() - st
    if save_predictions:
        os.makedirs(os.path.join(results_path, 'pred'), exist_ok=True)
        np.savetxt(os.path.join(results_path, 'pred', f'test_{algorithm_class.__name__}{ind}.csv'), model.predict(X_test), delimiter=",")
        np.savetxt(os.path.join(results_path, 'pred', f'train_{algorithm_class.__name__}{ind}.csv'), model.predict(X_train), delimiter=",")
    outp = params
    if 'estimator' in outp:
        outp['estimator'] = str(outp['estimator'])
    return {
        "algorithm": algorithm_class.__name__,
        "file_postfix": f"{algorithm_class.__name__}{ind}",
        "model_params": outp,
        "train_time_sec": tr_time,
        "inference_time_sec": inf_time,
        "train_accuracy": accuracy_score(model.predict(X_train), y_train),
        "test_accuracy": accuracy_score(model.predict(X_test), y_test)
    }

def load_algorithm(algorithm, algorithm_config, base_estimator_cfg, random_state):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    try:
        from xgboost import XGBClassifier
    except:
        XGBClassifier = None
    try:
        from catboost import CatBoostClassifier
    except:
        CatBoostClassifier = None
    try:
        from lightgbm import LGBMClassifier
    except:
        LGBMClassifier = None
    from models.brownboost import get_brownboost_class
    from models.madaboost import get_madaboost_class
    from models.filterboost import FilterBoostClassifier
    from sklearn.model_selection import ParameterGrid
    base_estimators = []
    if base_estimator_cfg['estimator_type'] == "stump":
        for p in ParameterGrid(base_estimator_cfg['estimator_params']):
            base_estimators.append(DecisionTreeClassifier(**p))
    param_grid = {}
    algorithm_class = None
    match algorithm:
        case "AdaBoost":
            algorithm_class = AdaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]
            param_grid["algorithm"] = ["SAMME"]
        case "GradientBoost":
            algorithm_class = GradientBoostingClassifier
            param_grid["loss"] = ["exponential"]
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]
        case "BrownBoost":
            from models.brownboost import BrownBoostClassifier, BrownBoostClassifierGPU
            gpu = algorithm_config["BrownBoost"].get("gpu", False)
            algorithm_class = BrownBoostClassifierGPU if gpu else BrownBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["c"] = algorithm_config["BrownBoost"]["c"]
            param_grid["convergence_criterion"] = algorithm_config["BrownBoost"]["convergence_criterion"]
            param_grid["max_estimators"] = [200000]
        case "MadaBoost":
            from models.madaboost import MadaBoostClassifier, MadaBoostClassifierGPU
            gpu = algorithm_config["MadaBoost"].get("gpu", False)
            algorithm_class = MadaBoostClassifierGPU if gpu else MadaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]
        case "FilterBoost":
            algorithm_class = FilterBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]
            param_grid["epsilon"] = algorithm_config["FilterBoost"]["epsilon"]
            param_grid["delta"] = algorithm_config["FilterBoost"]["delta"]
            param_grid["tau"] = algorithm_config["FilterBoost"]["tau"]
        case "CatBoost":
            if CatBoostClassifier:
                algorithm_class = CatBoostClassifier
                param_grid["allow_writing_files"] = [False]
                param_grid["silent"] = [True]
                param_grid["iterations"] = algorithm_config["common"]["n_estimators"]
                param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]
                param_grid["depth"] = algorithm_config["CatBoost"]["depth"]
                if algorithm_config["CatBoost"].get("gpu", False):
                    param_grid["task_type"] = ["GPU"]
        case "XGBoost":
            if XGBClassifier:
                algorithm_class = XGBClassifier
                param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
                param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]
                param_grid["max_depth"] = algorithm_config["XGBoost"]["max_depth"]
                param_grid["use_label_encoder"] = [False]
                param_grid["eval_metric"] = ["logloss"]
                param_grid["verbosity"] = [0]
                if algorithm_config["XGBoost"].get("gpu", False):
                    param_grid["tree_method"] = ["gpu_hist"]
        case "LightGBM":
            if LGBMClassifier:
                algorithm_class = LGBMClassifier
                param_grid["verbose"] = [-1]
                param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
                param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]
                param_grid["max_depth"] = algorithm_config["LightGBM"]["max_depth"]
                if algorithm_config["LightGBM"].get("gpu", False):
                    param_grid["device_type"] = ["gpu"]
    return (algorithm_class, param_grid)

class BoostingBenchmarkTrainer:
    def __init__(self, algorithms):
        self.algorithms = algorithms
    def fit_and_evaluate(self, X, y, random_state=None, test_size=0.15, results_path="results", test_name="test", multiprocessing=True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        os.makedirs(os.path.join(results_path, test_name), exist_ok=True)
        np.savetxt(os.path.join(results_path, test_name, 'train-dataset.csv'), np.column_stack((X_train, y_train)), delimiter=",")
        np.savetxt(os.path.join(results_path, test_name, 'test-dataset.csv'), np.column_stack((X_test, y_test)), delimiter=",")
        results = []
        if multiprocessing:
            c = cpu_count(logical=False)
            pool = Pool(processes=c)
            tasks = []
            for ac, pg in self.algorithms:
                if ac is None:
                    continue
                for ind, p in enumerate(ParameterGrid(pg)):
                    tasks.append(pool.apply_async(train_test_model, (ac, p, X_train, X_test, y_train, y_test, os.path.join(results_path, test_name)), {"ind": ind, "random_state": random_state}))
            for t in tasks:
                results.append(t.get())
            pool.close()
            pool.join()
        else:
            for ac, pg in self.algorithms:
                if ac is None:
                    continue
                for ind, p in enumerate(ParameterGrid(pg)):
                    results.append(train_test_model(ac, p, X_train, X_test, y_train, y_test, os.path.join(results_path, test_name), ind=ind, random_state=random_state))
        pd.DataFrame(results).to_csv(os.path.join(results_path, test_name, 'results.csv'), index=False, sep=",")
        return 0
