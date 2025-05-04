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
    base_estimators = []
    base_regressor_estimators = []

    match base_estimator_cfg['estimator_type']:
        case "stump":
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.tree import DecisionTreeRegressor
            for params in ParameterGrid(base_estimator_cfg['estimator_params']):
                base_estimators.append(DecisionTreeClassifier(**params))
                base_regressor_estimators.append(DecisionTreeRegressor(**params))
                
        case "neural_network":
            from neural_classifier import NeuralBinaryClassifier
            for params in ParameterGrid(base_estimator_cfg['estimator_params']):
                base_estimators.append(NeuralBinaryClassifier(**params))

    param_grid = dict()
    algorithm_class = None
    match algorithm:
        case "AdaBoost":
            from models.adaboost import AdaBoostClassifier
            algorithm_class = AdaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']

        case "GradientBoost":
            from models.gradientboost import (
                GradientBoostingClassifier, GradientBoostingClassifierGPU
            )
            gpu = algorithm_config["GradientBoost"].get("gpu", False)
            algorithm_class = GradientBoostingClassifierGPU if gpu else GradientBoostingClassifier
            param_grid["estimator"] = base_regressor_estimators
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["loss"] = algorithm_config["GradientBoost"]["loss"]

        case "BrownBoost":
            from models.brownboost import BrownBoostClassifier, BrownBoostClassifierGPU
            gpu = algorithm_config["BrownBoost"].get("gpu", False)
            algorithm_class = BrownBoostClassifierGPU if gpu else BrownBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["c"] = algorithm_config["BrownBoost"]["c"]
            param_grid["convergence_criterion"] = algorithm_config["BrownBoost"]["convergence_criterion"]
            param_grid["max_estimators"] = [ max(algorithm_config["common"]["n_estimators"])]
        
        case "MadaBoost":
            from models.madaboost import MadaBoostClassifier, MadaBoostClassifierGPU
            gpu = algorithm_config["MadaBoost"].get("gpu", False)
            algorithm_class = MadaBoostClassifierGPU if gpu else MadaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "RealAdaBoost":
            from models.realadaboost import RealAdaBoostClassifier
            algorithm_class = RealAdaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "ModestAdaBoost":
            from models.modestadaboost import ModestAdaBoostClassifier
            algorithm_class = ModestAdaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "LogitBoost":
            from models.logitboost import LogitBoost
            algorithm_class = LogitBoost
            param_grid["estimator"] = base_regressor_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "SOWAdaBoost":
            from models.sowadaboost import SOWAdaBoostClassifier, SOWAdaBoostClassifierGPU
            gpu = algorithm_config["SOWAdaBoost"].get("gpu", False)
            algorithm_class = SOWAdaBoostClassifierGPU if gpu else SOWAdaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "XWaterBoost":
            from models.waterboost import XWaterBoostClassifier
            algorithm_class = XWaterBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "XMWaterBoost":
            from models.waterboost import XMWaterBoostClassifier
            algorithm_class = XMWaterBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "MWaterBoost":
            from models.waterboost import MWaterBoostClassifier
            algorithm_class = MWaterBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "AWaterBoost":
            from models.waterboost import AWaterBoostClassifier
            algorithm_class = AWaterBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "XMadaBoost":
            from models.waterboost import XMadaBoostClassifier
            algorithm_class = XMadaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "FilterBoost":
            from models.filterboost import FilterBoostClassifier
            algorithm_class = FilterBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["epsilon"] = algorithm_config['FilterBoost']['epsilon']
            param_grid["delta"] = algorithm_config['FilterBoost']['delta']
            param_grid["tau"] = algorithm_config['FilterBoost']['tau']

        # No estimator for CatBoost
        case "CatBoost":
            from catboost import CatBoostClassifier
            algorithm_class = CatBoostClassifier
            param_grid['allow_writing_files'] = [False]
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

        case "NGBoost":
            from models.ngboost import NGBoostClassifier
            algorithm_class = NGBoostClassifier
            param_grid["estimator"] = base_regressor_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "SMOTEBoost":
            from models.smoteboost import SMOTEBoostClassifier
            algorithm_class = SMOTEBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]
        
        case "RUSBoost":
            from imblearn.ensemble import RUSBoostClassifier
            algorithm_class = RUSBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]
        
        case _:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    return (algorithm_class, param_grid)


class BoostingBenchmarkTrainer:
    def __init__(self, algorithms_data):
        self.algorithms_data = algorithms_data

    def fit_and_evaluate(self, X, y, random_state=None, test_size=0.15, results_path="results", test_name="test", multiprocessing=True):
        results = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size, stratify=y)   
        print(f"== Starting {test_name} ==")
        os.mkdir(os.path.join(results_path,test_name))
        np.savetxt(os.path.join(results_path,test_name,'train-dataset.csv'), np.hstack((X_train, y_train.reshape(X_train.shape[0], 1))), delimiter=",")
        np.savetxt(os.path.join(results_path,test_name,'test-dataset.csv'), np.hstack((X_test, y_test.reshape(X_test.shape[0], 1))), delimiter=",")
        save_predictions = ('random' not in test_name)
        if save_predictions:
            os.mkdir(os.path.join(results_path, test_name, 'pred'))
        if multiprocessing:
            cpus = cpu_count(logical=False)
            if type(multiprocessing) is int:
                cpus = multiprocessing
            with Pool(processes=cpus) as pool:
                tasks = []
                for algorithm_class, algorithm_param_grid in self.algorithms_data:
                    print(f"= Testing {algorithm_class.__name__} =")
                    for ind, params in enumerate(ParameterGrid(algorithm_param_grid)):
                        tasks.append(pool.apply_async(train_test_model, args=[algorithm_class, params, X_train, X_test, y_train, y_test, os.path.join(results_path, test_name)], kwds={"ind" : ind, "random_state" : random_state, "save_predictions" : save_predictions}))           
                for t in tasks:
                    results.append(t.get(timeout=None))
        else:
            for algorithm_class, algorithm_param_grid in self.algorithms_data:
                print(f"= Testing {algorithm_class.__name__} =")
                for ind, params in enumerate(ParameterGrid(algorithm_param_grid)):
                    results.append(train_test_model(algorithm_class, params, X_train, X_test, y_train, y_test, os.path.join(results_path, test_name), ind=ind, random_state=random_state, save_predictions=save_predictions))
        
        print("= Writing results =")
        pd.DataFrame(results).to_csv(os.path.join(results_path, test_name, 'results.csv'), index=False, sep=",")
        print(f"== Finished {test_name} ==")
        return 0
