import numpy as np
import pandas as pd
import json
import os
import time
from psutil import cpu_count
from sklearn.model_selection import train_test_split, ParameterGrid
from multiprocessing import Pool
from sklearn.metrics import accuracy_score

def train_test_model(algorithm_class, params, X, y, results_path,  N_retrain=1, ind="", random_state=None, save_predictions=True, feature_noise=0, label_noise=0, test_size=0.15, validation_size=0.15):
    rng = np.random.default_rng(seed=random_state)
    classes = np.sort(np.unique(y))
    class_proportions = [np.where(y==classes[0], 1, 0).sum() / y.shape[0], np.where(y==classes[1], 1, 0).sum() / y.shape[0]]
    major_class = np.argmax(class_proportions)
    minor_class = np.argmin(class_proportions)

    tr_time_sum = 0
    inf_time_sum = 0
    unweighted_train_accuracy_sum = 0
    unweighted_validation_accuracy_sum = 0
    unweighted_test_accuracy_sum = 0
    train_accuracy_sum = 0
    validation_accuracy_sum = 0
    test_accuracy_sum = 0
    minor_class_test_accuracy_sum = 0
    major_class_test_accuracy_sum = 0

    unweighted_train_accuracy_bestmodel = 0
    unweighted_validation_accuracy_bestmodel = 0
    unweighted_test_accuracy_bestmodel = 0
    train_accuracy_bestmodel = 0
    validation_accuracy_bestmodel = 0
    test_accuracy_bestmodel = 0
    minor_class_test_accuracy_bestmodel = 0
    major_class_test_accuracy_bestmodel = 0
    tr_time_bestmodel  = 0
    inf_time_bestmodel = 0

    unweighted_train_accuracy_worstmodel = 0
    unweighted_validation_accuracy_worstmodel = 0
    unweighted_test_accuracy_worstmodel = 0
    train_accuracy_worstmodel = 0
    validation_accuracy_worstmodel = 0
    test_accuracy_worstmodel = 1
    minor_class_test_accuracy_worstmodel = 0
    major_class_test_accuracy_worstmodel = 0
    tr_time_worstmodel = 0
    inf_time_worstmodel = 0

    inf_time_bestvalidmodel = 0
    tr_time_bestvalidmodel = 0
    unweighted_train_accuracy_bestvalidmodel = 0
    unweighted_validation_accuracy_bestvalidmodel = 0
    unweighted_test_accuracy_bestvalidmodel = 0
    train_accuracy_bestvalidmodel = 0
    validation_accuracy_bestvalidmodel = 0
    test_accuracy_bestvalidmodel = 0
    minor_class_test_accuracy_bestvalidmodel = 0
    major_class_test_accuracy_bestvalidmodel = 0
    

    all_pred = np.zeros(y.shape)
    for _ in range(N_retrain):
        X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, random_state=rng.integers(0, 1000, size=1)[0], test_size=test_size, stratify=y)   
        X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, random_state=rng.integers(0, 1000, size=1)[0], test_size=validation_size/(1-test_size), stratify=y_train_validation)
        
        #feature_noise
        X *= (np.ones(X.shape)+rng.normal(0, feature_noise, X.shape))
        
        # label noise
        y_train = np.where(rng.random(X_train.shape[0]) < label_noise, 1-y_train,  y_train)
        y_validation = np.where(rng.random(X_validation.shape[0]) < label_noise, 1-y_validation, y_validation)
    
        train_class_weights = np.where(y_train==major_class, 1, class_proportions[major_class]/class_proportions[minor_class])
        train_class_weights /= train_class_weights.sum()

        validation_class_weights = np.where(y_validation==major_class, 1, class_proportions[major_class]/class_proportions[minor_class])
        validation_class_weights /= validation_class_weights.sum()

        test_class_weights = np.where(y_test==major_class, 1, class_proportions[major_class]/class_proportions[minor_class])
        test_class_weights /= test_class_weights.sum()

        minor_class_test_ind = np.where(y_test==classes[minor_class])
        major_class_test_ind = np.where(y_test==classes[major_class])  

        model = algorithm_class(**params)
        st = time.time()
        model.fit(X_train, y_train)
        tr_time = time.time() - st
        st = time.time()
        model.predict(X_test)
        inf_time = time.time() - st

        

        unweighted_train_accuracy = accuracy_score(model.predict(X_train), y_train)
        unweighted_validation_accuracy = accuracy_score(model.predict(X_validation), y_validation)
        unweighted_test_accuracy = accuracy_score(model.predict(X_test), y_test)
        train_accuracy = accuracy_score(model.predict(X_train), y_train, sample_weight=train_class_weights)
        validation_accuracy = accuracy_score(model.predict(X_validation), y_validation, sample_weight=validation_class_weights)
        test_accuracy = accuracy_score(model.predict(X_test), y_test, sample_weight=test_class_weights)
        minor_class_test_accuracy = accuracy_score(model.predict(X_test)[minor_class_test_ind], y_test[minor_class_test_ind])
        major_class_test_accuracy = accuracy_score(model.predict(X_test)[major_class_test_ind], y_test[major_class_test_ind])

        if test_accuracy > test_accuracy_bestmodel:
            all_pred_bestmodel = model.predict(X)
            inf_time_bestmodel = inf_time
            tr_time_bestmodel = tr_time
            unweighted_train_accuracy_bestmodel = unweighted_train_accuracy
            unweighted_validation_accuracy_bestmodel = unweighted_validation_accuracy
            unweighted_test_accuracy_bestmodel = unweighted_test_accuracy
            train_accuracy_bestmodel = train_accuracy
            validation_accuracy_bestmodel = validation_accuracy
            test_accuracy_bestmodel = test_accuracy
            minor_class_test_accuracy_bestmodel = minor_class_test_accuracy
            major_class_test_accuracy_bestmodel = major_class_test_accuracy

        if test_accuracy < test_accuracy_worstmodel:
            inf_time_worstmodel = inf_time
            tr_time_worstmodel = tr_time
            unweighted_train_accuracy_worstmodel = unweighted_train_accuracy
            unweighted_validation_accuracy_worstmodel = unweighted_validation_accuracy
            unweighted_test_accuracy_worstmodel = unweighted_test_accuracy
            train_accuracy_worstmodel = train_accuracy
            validation_accuracy_worstmodel = validation_accuracy
            test_accuracy_worstmodel = test_accuracy
            minor_class_test_accuracy_worstmodel = minor_class_test_accuracy
            major_class_test_accuracy_worstmodel = major_class_test_accuracy

        tr_time_sum += tr_time
        inf_time_sum += inf_time
        unweighted_train_accuracy_sum += unweighted_train_accuracy
        unweighted_validation_accuracy_sum += unweighted_validation_accuracy
        unweighted_test_accuracy_sum += unweighted_test_accuracy
        train_accuracy_sum += train_accuracy
        validation_accuracy_sum += validation_accuracy
        test_accuracy_sum += test_accuracy
        minor_class_test_accuracy_sum += minor_class_test_accuracy
        major_class_test_accuracy_sum += major_class_test_accuracy



        if validation_accuracy > validation_accuracy_bestvalidmodel:
            inf_time_bestvalidmodel = inf_time
            tr_time_bestvalidmodel = tr_time
            unweighted_train_accuracy_bestvalidmodel = unweighted_train_accuracy
            unweighted_validation_accuracy_bestvalidmodel = unweighted_validation_accuracy
            unweighted_test_accuracy_bestvalidmodel = unweighted_test_accuracy
            train_accuracy_bestvalidmodel = train_accuracy
            validation_accuracy_bestvalidmodel = validation_accuracy
            test_accuracy_bestvalidmodel = test_accuracy
            minor_class_test_accuracy_bestvalidmodel = minor_class_test_accuracy
            major_class_test_accuracy_bestvalidmodel = major_class_test_accuracy


    if save_predictions:
        os.makedirs(os.path.join(results_path, 'pred'), exist_ok=True)
        np.savetxt(os.path.join(results_path, 'pred', f'{algorithm_class.__name__}{ind}_pred.csv'), all_pred_bestmodel, delimiter=",")

    outp = params
    if 'estimator' in outp:
        outp['estimator'] = str(outp['estimator'])
    return {
        "algorithm": algorithm_class.__name__,
        "file_postfix": f"{algorithm_class.__name__}{ind}",
        "model_params": outp,
        "mean_results_train_time_sec": tr_time_sum/N_retrain,
        "mean_results_inference_time_sec": inf_time_sum/N_retrain,
        "mean_results_unweighted_train_accuracy": unweighted_train_accuracy_sum/N_retrain,
        "mean_results_unweighted_validation_accuracy": unweighted_validation_accuracy_sum/N_retrain,
        "mean_results_unweighted_test_accuracy": unweighted_test_accuracy_sum/N_retrain,
        "mean_results_train_accuracy": train_accuracy_sum/N_retrain,
        "mean_results_validation_accuracy" : validation_accuracy_sum/N_retrain,
        "mean_results_test_accuracy": test_accuracy_sum/N_retrain,
        "mean_results_minor_class_test_accuracy": minor_class_test_accuracy_sum/N_retrain,
        "mean_results_major_class_test_accuracy": major_class_test_accuracy_sum/N_retrain,
        "best_results_train_time_sec": tr_time_bestmodel,
        "best_results_inference_time_sec": inf_time_bestmodel,
        "best_results_unweighted_train_accuracy": unweighted_train_accuracy_bestmodel,
        "best_results_unweighted_validation_accuracy": unweighted_validation_accuracy_bestmodel,
        "best_results_unweighted_test_accuracy": unweighted_test_accuracy_bestmodel,
        "best_results_train_accuracy": train_accuracy_bestmodel,
        "best_results_validation_accuracy" : validation_accuracy_bestmodel,
        "best_results_test_accuracy": test_accuracy_bestmodel,
        "best_results_minor_class_test_accuracy": minor_class_test_accuracy_bestmodel,
        "best_results_major_class_test_accuracy": major_class_test_accuracy_bestmodel,
        "bestvalid_results_train_time_sec": tr_time_bestvalidmodel,
        "bestvalid_results_inference_time_sec": inf_time_bestvalidmodel,
        "bestvalid_results_unweighted_train_accuracy": unweighted_train_accuracy_bestvalidmodel,
        "bestvalid_results_unweighted_validation_accuracy": unweighted_validation_accuracy_bestvalidmodel,
        "bestvalid_results_unweighted_test_accuracy": unweighted_test_accuracy_bestvalidmodel,
        "bestvalid_results_train_accuracy": train_accuracy_bestvalidmodel,
        "bestvalid_results_validation_accuracy" : validation_accuracy_bestvalidmodel,
        "bestvalid_results_test_accuracy": test_accuracy_bestvalidmodel,
        "bestvalid_results_minor_class_test_accuracy": minor_class_test_accuracy_bestvalidmodel,
        "bestvalid_results_major_class_test_accuracy": major_class_test_accuracy_bestvalidmodel,
        "worst_results_train_time_sec": tr_time_worstmodel,
        "worst_results_inference_time_sec": inf_time_worstmodel,
        "worst_results_unweighted_train_accuracy": unweighted_train_accuracy_worstmodel,
        "worst_results_unweighted_validation_accuracy": unweighted_validation_accuracy_worstmodel,
        "worst_results_unweighted_test_accuracy": unweighted_test_accuracy_worstmodel,
        "worst_results_train_accuracy": train_accuracy_worstmodel,
        "worst_results_validation_accuracy" : validation_accuracy_worstmodel,
        "worst_results_test_accuracy": test_accuracy_worstmodel,
        "worst_results_minor_class_test_accuracy": minor_class_test_accuracy_worstmodel,
        "worst_results_major_class_test_accuracy": major_class_test_accuracy_worstmodel,
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
            from models.gradientboost import GradientBoostingClassifier
            algorithm_class = GradientBoostingClassifier
            param_grid["estimator"] = base_regressor_estimators
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["loss"] = algorithm_config["GradientBoost"]["loss"]

        case "BrownBoost":
            from models.brownboost import BrownBoostClassifier
            algorithm_class = BrownBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["c"] = algorithm_config["BrownBoost"]["c"]
            param_grid["convergence_criterion"] = algorithm_config["BrownBoost"]["convergence_criterion"]
            param_grid["n_estimators"] = [ max(algorithm_config["common"]["n_estimators"])]
        
        case "MadaBoost":
            from models.madaboost import MadaBoostClassifier
            algorithm_class = MadaBoostClassifier
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
            from models.sowadaboost import SOWAdaBoostClassifier
            algorithm_class = SOWAdaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "WaterBoost":
            from models.waterboost import WaterBoostClassifier
            algorithm_class = WaterBoostClassifier
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
            from models.rusboost import RUSBoostClassifier
            algorithm_class = RUSBoostClassifier
            param_grid["estimator"]      = base_estimators
            param_grid["n_estimators"]   = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"]  = algorithm_config["common"]["learning_rate"]
        
        case _:
            raise ValueError(f"Unknown algorithm: {algorithm}")


    return (algorithm_class, param_grid)


class BoostingBenchmarkTrainer:
    def __init__(self, algorithms_data):
        self.algorithms_data = algorithms_data

    def fit_and_evaluate(self, X, y, random_state=None, test_size=0.15, validation_size=0.15, N_retrain=1, results_path="results", test_name="test", multiprocessing=True, label_noise=0, feature_noise=0):
        results = []

        print(f"== Starting {test_name} ==")
        os.makedirs(os.path.join(results_path,test_name), exist_ok=True)
        #np.savetxt(os.path.join(results_path,test_name,'train-dataset.csv'), np.hstack((X_train, y_train.reshape(X_train.shape[0], 1))), delimiter=",")
        #np.savetxt(os.path.join(results_path,test_name,'test-dataset.csv'), np.hstack((X_test, y_test.reshape(X_test.shape[0], 1))), delimiter=",")
        #np.savetxt(os.path.join(results_path,test_name,'validation-dataset.csv'), np.hstack((X_validation, y_validation.reshape(X_validation.shape[0], 1))), delimiter=",")

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
                        tasks.append(pool.apply_async(train_test_model, args=[algorithm_class, params, X, y, os.path.join(results_path, test_name)], kwds={"ind" : ind, "random_state" : random_state, "save_predictions" : save_predictions, "N_retrain" : N_retrain, 'label_noise' : label_noise, "feature_noise" : feature_noise, "validation_size" : validation_size, "test_size" : test_size}))           
                for t in tasks:
                    results.append(t.get(timeout=None))
        else:
            for algorithm_class, algorithm_param_grid in self.algorithms_data:
                print(f"= Testing {algorithm_class.__name__} =")
                for ind, params in enumerate(ParameterGrid(algorithm_param_grid)):
                    results.append(train_test_model(algorithm_class, params, X, y, os.path.join(results_path, test_name), ind=ind, random_state=random_state, save_predictions=save_predictions,  N_retrain=N_retrain, label_noise=label_noise, feature_noise=feature_noise, test_size=test_size, validation_size=validation_size))
        
        print("= Writing results =")
        pd.DataFrame(results).to_csv(os.path.join(results_path, test_name, 'results.csv'), index=False, sep=",")
        print(f"== Finished {test_name} ==")
        return 0
