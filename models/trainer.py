import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

from .model_configs import get_memory_usage_mb


class GBMBenchmarkTrainer:
    def __init__(
            self,
            model_configs: dict,
            task_type: str = 'classification'
    ):
        self.model_configs = model_configs
        self.task_type = task_type

        if self.task_type == 'classification':
            self.metric_name = "accuracy"
            self.metric_func = accuracy_score
        else:
            self.metric_name = "mse"
            self.metric_func = mean_squared_error

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        results = []
        trained_models = {}

        for model_name, cfg in self.model_configs.items():
            model_class = cfg["model_class"]
            model_params = cfg["params"]

            model = model_class(**model_params)

            mem_before = get_memory_usage_mb()
            time_before = time.time()

            model.fit(X_train, y_train)

            train_time = time.time() - time_before
            mem_after = get_memory_usage_mb()
            mem_usage = mem_after - mem_before

            time_before = time.time()
            preds = model.predict(X_test)
            inference_time = time.time() - time_before

            metric_value = self.metric_func(y_test, preds)

            trained_models[model_name] = model

            results.append({
                "model": model_name,
                "train_time_sec": train_time,
                "inference_time_sec": inference_time,
                "memory_usage_mb": mem_usage,
                self.metric_name: metric_value
            })

        results_df = pd.DataFrame(results)
        return results_df, trained_models
