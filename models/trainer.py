import time
import numpy as np
import json
from sklearn.metrics import accuracy_score
from utils.misc import get_memory_usage_mb
from sklearn.model_selection import GridSearchCV

def load_algorithm(algorithm, algorithm_config, base_estimator_cfg):
    # base estimator initialization
    match base_estimator_cfg['estimator_type']:
        case "stump":
            from sklearn.tree import DecisionTreeClassifier
            base_estimator = DecisionTreeClassifier(**base_estimator_cfg['estimator_params'])
        case "neural_network":
            pass

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

        case "GradientBoosting":
            from sklearn.ensemble import GradientBoostingClassifier
            algorithm_class = GradientBoostingClassifier
            param_grid["loss"] = ['exponential']
            param_grid["estimator"] = [base_estimator]
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
    return (algorithm_class, param_grid)



class BoostingBenchmarkTrainer:
    def __init__(self, base_estimator_cfg, algorithms : list, algorithm_configs: dict):
        self.base_estimator_cfg = base_estimator_cfg
        self.algorithms = algorithms
        self.algorithm_configs = algorithm_configs
        
    def fit_and_evaluate(self, X_train, X_test, y_train, y_test, test_name="test"):
        results = {}
        for algorithm_name in self.algorithms:

            algorithm_class, algorithm_param_grid = load_algorithm(algorithm=algorithm_name, algorithm_config=self.algorithm_configs, base_estimator_cfg=self.base_estimator_cfg)

            mem_before = get_memory_usage_mb()
            model = GridSearchCV(algorithm_class(), param_grid=algorithm_param_grid, n_jobs=-1)


            time_before = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - time_before

            mem_after = get_memory_usage_mb()
            mem_usage = mem_after - mem_before

            time_before = time.time()
            preds = model.best_estimator_.predict(X_test)
            inference_time = time.time() - time_before

            np.save(f'{test_name}_{algorithm_name}', preds)

            results[algorithm_name] = {
                "model_params" : str(model.best_params_),
                "train_time_sec": train_time,
                "inference_time_sec": inference_time,
                "memory_usage_mb": mem_usage,
                "train_accuracy" : accuracy_score(model.predict(X_train), y_train),
                "test_accuracy" : accuracy_score(model.predict(X_test), y_test)
            }

        with open(f'{test_name}_results', 'w') as file:
            json.dump(results, file)
        return 0
