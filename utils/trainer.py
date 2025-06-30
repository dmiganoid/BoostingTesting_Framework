import numpy as np
import pandas as pd
import json
import os
import time
from psutil import cpu_count
from sklearn.model_selection import train_test_split, ParameterGrid
from multiprocessing import Pool
from collections.abc import Callable

def train_test_model(algorithm_class, params, X, y, results_path, metric_score : Callable[[float, float], float], use_class_weights=False, N_retrain=1, ind="", random_state=None, save_predictions=True, feature_noise=0, label_noise=0, test_size=0.15, validation_size=0.15):
    """Train and evaluate a boosting model with specified parameters.

    Fits the model on training data, evaluates performance on train, validation, and test sets,
    and tracks best/worst/mean metrics across retraining iterations. On each iterations supports noise injection and
    class-weighted metrics. Saves predictions if specified.

    Args:
        algorithm_class (class): Boosting algorithm class to instantiate.
        params (dict): Hyperparameters for the algorithm.
        X (array-like): Feature matrix for training and evaluation.
        y (array-like): Target labels for training and evaluation.
        results_path (str): Directory to save results and predictions.
        metric_score (Callable): Function to compute evaluation metric, taking true and predicted values.
        use_class_weights (bool, optional): Whether to use class weights in metrics. Defaults to False.
        N_retrain (int, optional): Number of retraining iterations. Defaults to 1.
        ind (str, optional): Index or identifier for the model run. Defaults to "".
        random_state (int, optional): Random seed for reproducibility. Defaults to None.
        save_predictions (bool, optional): Whether to save model predictions. Defaults to True.
        feature_noise (float, optional): Proportion of feature noise to inject. Defaults to 0.
        label_noise (float, optional): Proportion of label noise to inject. Defaults to 0.
        test_size (float, optional): Proportion of data for test split. Defaults to 0.15.
        validation_size (float, optional): Proportion of data for validation split. Defaults to 0.15.

    Returns:
        dict: Dictionary containing detailed performance metrics and metadata with the following structure:
            - 'algorithm' (str): Name of the algorithm class (e.g., 'AdaBoostClassifier').
            - 'file_postfix' (str): Identifier for the model run, combining algorithm name and index (e.g., 'AdaBoostClassifier0').
            - 'params' (dict): Hyperparameters used for the model, with 'estimator' converted to string if present.
            - 'mean_results' (dict): Averaged metrics over N_retrain iterations, including:
                - 'tr_time' (float): Average training time.
                - 'inf_time' (float): Average inference time.
                - 'train_metric' (float): Average metric score on training set.
                - 'validation_metric' (float): Average metric score on validation set.
                - 'test_metric' (float): Average metric score on test set.
                - 'minor_class_test_metric' (float): Average metric score for the minority class on test set.
                - 'major_class_test_metric' (float): Average metric score for the majority class on test set.
            - 'best_test_metric_results' (dict): Metrics for the iteration with the highest test set metric, with the same keys as 'mean_results'.
            - 'worst_test_metric_results' (dict): Metrics for the iteration with the lowest test set metric, with the same keys as 'mean_results'.
            - 'best_validation_metric_results' (dict): Metrics for the iteration with the highest validation set metric, with the same keys as 'mean_results'.
    """
    
    rng = np.random.default_rng(seed=random_state)

    mean_results = {
        "tr_time" : 0,
        "inf_time" : 0,
        "train_metric" : 0,
        "validation_metric" : 0,
        "test_metric" : 0,
        "train_metric" : 0,
        "validation_metric" : 0,
        "test_metric" : 0,
        "minor_class_test_metric" : 0,
        "major_class_test_metric" : 0
    }

    best_test_metric_results = {    
        "train_metric" : 0,
        "validation_metric" : 0,
        "test_metric" : 0,
        "train_metric" : 0,
        "validation_metric" : 0,
        "test_metric" : 0,
        "minor_class_test_metric" : 0,
        "major_class_test_metric" : 0,
        "tr_time"  : 0,
        "inf_time" : 0,
    }

    worst_test_metric_results = {
        "train_metric" : 0,
        "validation_metric" : 0,
        "test_metric" : 1,
        "train_metric" : 0,
        "validation_metric" : 0,
        "test_metric" : np.inf,
        "minor_class_test_metric" : 0,
        "major_class_test_metric" : 0,
        "tr_time" : 0,
        "inf_time" : 0
    }

    best_validation_metric_results = {
        "inf_time" : 0,
        "tr_time" : 0,
        "train_metric" : 0,
        "validation_metric" : 0,
        "test_metric" : 0,
        "train_metric" : 0,
        "validation_metric" : 0,
        "test_metric" : 0,
        "minor_class_test_metric" : 0,
        "major_class_test_metric" : 0
    }

    # class weights for weighted metrics
    classes = np.sort(np.unique(y))
    class_proportions = [np.where(y==classes[0], 1, 0).sum() / y.shape[0], np.where(y==classes[1], 1, 0).sum() / y.shape[0]]
    major_class = np.argmax(class_proportions)
    minor_class = np.argmin(class_proportions)

    for _ in range(N_retrain):
        X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, random_state=rng.integers(0, 1000, size=1)[0], test_size=test_size, stratify=y)   
        X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, random_state=rng.integers(0, 1000, size=1)[0], test_size=validation_size/(1-test_size), stratify=y_train_validation)
        
        # feature noise
        if feature_noise != 0:
            X *= (np.ones(X.shape)+rng.normal(0, feature_noise, X.shape))
        
        # label noise
        if label_noise != 0:
            y_train = np.where(rng.random(X_train.shape[0]) < label_noise, 1-y_train,  y_train)
            y_validation = np.where(rng.random(X_validation.shape[0]) < label_noise, 1-y_validation, y_validation)
    
        if use_class_weights: # set weights for each object
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

        if use_class_weights:
            train_metric = metric_score(model.predict(X_train), y_train, sample_weight=train_class_weights)
            validation_metric = metric_score(model.predict(X_validation), y_validation, sample_weight=validation_class_weights)
            test_metric = metric_score(model.predict(X_test), y_test, sample_weight=test_class_weights)
        else:
            train_metric = metric_score(model.predict(X_train), y_train)
            validation_metric = metric_score(model.predict(X_validation), y_validation)
            test_metric = metric_score(model.predict(X_test), y_test)
        
        minor_class_test_metric = metric_score(model.predict(X_test)[minor_class_test_ind], y_test[minor_class_test_ind])
        major_class_test_metric = metric_score(model.predict(X_test)[major_class_test_ind], y_test[major_class_test_ind])

        mean_results["inf_time"] += inf_time
        mean_results["tr_time"] += tr_time
        mean_results["train_metric"] += train_metric
        mean_results["validation_metric"] += validation_metric
        mean_results["test_metric"] += test_metric
        mean_results["minor_class_test_metric"] += minor_class_test_metric
        mean_results["major_class_test_metric"] += major_class_test_metric
    
        if test_metric > best_test_metric_results['test_metric']:
            best_test_metric_predictions = model.predict(X)
            best_test_metric_results["inf_time"] = inf_time
            best_test_metric_results["tr_time"] = tr_time
            best_test_metric_results["train_metric"] = train_metric
            best_test_metric_results["validation_metric"] = validation_metric
            best_test_metric_results["test_metric"] = test_metric
            best_test_metric_results["minor_class_test_metric"] = minor_class_test_metric
            best_test_metric_results["major_class_test_metric"] = major_class_test_metric

        if test_metric < worst_test_metric_results["test_metric"]:
            worst_test_metric_results["inf_time"] = inf_time
            worst_test_metric_results["tr_time"] = tr_time
            worst_test_metric_results["train_metric"] = train_metric
            worst_test_metric_results["validation_metric"] = validation_metric
            worst_test_metric_results["test_metric"] = test_metric
            worst_test_metric_results["minor_class_test_metric"] = minor_class_test_metric
            worst_test_metric_results["major_class_test_metric"] = major_class_test_metric

        if validation_metric > best_validation_metric_results["validation_metric"]:
            best_validation_metric_results["inf_time"] = inf_time
            best_validation_metric_results["tr_time"] = tr_time
            best_validation_metric_results["train_metric"] = train_metric
            best_validation_metric_results["validation_metric"] = validation_metric
            best_validation_metric_results["test_metric"] = test_metric
            best_validation_metric_results["minor_class_test_metric"] = minor_class_test_metric
            best_validation_metric_results["major_class_test_metric"] = major_class_test_metric


    if save_predictions:
        os.makedirs(os.path.join(results_path, 'pred'), exist_ok=True)
        np.savetxt(os.path.join(results_path, 'pred', f'{algorithm_class.__name__}{ind}_pred.csv'), best_test_metric_predictions, delimiter=",")

    
    outp = params 
    if 'estimator' in outp:
        outp['estimator'] = str(outp['estimator'])
    return {
        "algorithm": algorithm_class.__name__,
        "file_postfix": f"{algorithm_class.__name__}{ind}",
        "params": outp,
        "mean_results" : mean_results,
        "best_validation_metric_results" : best_validation_metric_results,
        "best_test_metric_results" : best_test_metric_results,
        "worst_test_metric_results" : worst_test_metric_results
    }


class BoostingBenchmarkTrainer:
    def __init__(self, algorithms_data):
        """Initialize BoostingBenchmarkTrainer with algorithm configurations.

        Args:
            algorithms_data (list): List of tuples containing algorithm classes and their parameter grids.
        """

        self.algorithms_data = algorithms_data

    def fit_and_evaluate(self, X, y, metric_function : Callable[[float, float], float], use_class_weights : bool=False, random_state=None, test_size=0.15, validation_size=0.15, N_retrain=1, results_path="results", test_name="test", multiprocessing=True, label_noise=0, feature_noise=0):
        """Train and evaluate performance of boosting algorithms on provided data.

        Fits algorithms on input data, evaluates performance using a specified metric, and saves results.
        Supports multiprocessing for parallel execution and noise injection in data.

        Args:
            X (array-like): Feature matrix for training and evaluation.
            y (array-like): Target labels for training and evaluation.
            metric_function (Callable): Function to analize performance, taking true and predicted values.
            use_class_weights (bool, optional): Whether to use class weights in metric computation. Weights are defined by class proportions.  Defaults to False.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
            test_size (float, optional): Proportion of data for test split. Defaults to 0.15.
            validation_size (float, optional): Proportion of data for validation split. Defaults to 0.15.
            N_retrain (int, optional): Number of retraining iterations. Defaults to 1.
            results_path (str, optional): Directory to save results. Defaults to "results".
            test_name (str, optional): Name of the test for result organization. Defaults to "test".
            multiprocessing (bool or int, optional): Enable multiprocessing or specify number of CPUs. Defaults to True.
            label_noise (float, optional): Proportion of class labels to be flipped. Defaults to 0.
            feature_noise (float, optional):  MSE of Relative error for features under normal distribution. Defaults to 0.

        Returns:
            int: Returns 0 upon successful completion.
        """

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
                        tasks.append(pool.apply_async(train_test_model, args=[algorithm_class, params, X, y, os.path.join(results_path, test_name)], kwds={"ind" : ind, "metric_score" : metric_function, "use_class_weights" : use_class_weights, "random_state" : random_state, "save_predictions" : save_predictions, "N_retrain" : N_retrain, 'label_noise' : label_noise, "feature_noise" : feature_noise, "validation_size" : validation_size, "test_size" : test_size}))           
                for t in tasks:
                    results.append(t.get(timeout=None))
        else:
            for algorithm_class, algorithm_param_grid in self.algorithms_data:
                print(f"= Testing {algorithm_class.__name__} =")
                for ind, params in enumerate(ParameterGrid(algorithm_param_grid)):
                    results.append(train_test_model(algorithm_class, params, X, y, os.path.join(results_path, test_name), ind=ind, random_state=random_state, metric_score=metric_function, use_class_weights=use_class_weights, save_predictions=save_predictions,  N_retrain=N_retrain, label_noise=label_noise, feature_noise=feature_noise, test_size=test_size, validation_size=validation_size))
        
        print("= Writing results =")
        pd.DataFrame(results).to_csv(os.path.join(results_path, test_name, 'results.csv'), index=False, sep=",")
        print(f"== Finished {test_name} ==")
        return 0
