import time
import numpy as np
import json
from sklearn.metrics import accuracy_score
from utils.misc import get_memory_usage_mb
from sklearn.model_selection import GridSearchCV

class BoostingBenchmarkTrainer:
    def __init__(self, base_estimator, algorithms : list, algorithm_configs: dict):
        self.base_estimator = base_estimator
        self.algorithms = algorithms
        self.algorithm_configs = algorithm_configs
        
    def fit_and_evaluate(self, X_train, y_train, X_test, y_test, test_name="test"):
        results = {}
        trained_models = {}

        for algorithm in self.algorithms:
            algorithm_name = type(algorithm).__name__
            algorithm_param_grid = self.algorithm_configs['common']
            if algorithm_name in self.algorithm_configs['common'].keys():
                for key,value in self.algorithm_configs['common'][algorithm_name].items():
                    algorithm_param_grid[key] = [value]

            print(algorithm)
            mem_before = get_memory_usage_mb()
            model = GridSearchCV(algorithm, param_grid=algorithm_param_grid)

            time_before = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - time_before

            mem_after = get_memory_usage_mb()
            mem_usage = mem_after - mem_before

            time_before = time.time()
            preds = model.best_estimator_.predict(X_test)
            inference_time = time.time() - time_before

            np.save(f'results/{test_name}_{algorithm_name}.npy', preds)

            trained_models[algorithm_name] = model

            results[algorithm_name] = {
                "model_params" : model.best_params_,
                "train_time_sec": train_time,
                "inference_time_sec": inference_time,
                "memory_usage_mb": mem_usage,
                "train_accuracy" : accuracy_score(model.predict(X_train), y_train),
                "test_accuracy" : accuracy_score(model.predict(X_test), y_test)
            }

        json.dump(results, f'results/test_{test_name}_results')
        return 0
