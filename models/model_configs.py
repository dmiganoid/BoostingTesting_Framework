# gbm_benchmark/models/model_configs.py

import os
import psutil

# Scikit-learn
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

# XGBoost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    xgb = None
    XGBClassifier = None
    XGBRegressor = None

# LightGBM
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    lgb = None
    LGBMClassifier = None
    LGBMRegressor = None

# CatBoost
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None


def get_default_model_configs(task_type='classification', random_state=42):
    """
    Возвращает словарь с базовыми конфигурациями моделей для бустинга.
    Если в окружении отсутствуют некоторые библиотеки, модель не включается.
    """
    if task_type not in ('classification', 'regression'):
        raise ValueError("task_type должен быть 'classification' или 'regression'.")

    if task_type == 'classification':
        gbm_class = GradientBoostingClassifier
        ada_class = AdaBoostClassifier
        xgb_class = XGBClassifier
        lgb_class = LGBMClassifier
        cat_class = CatBoostClassifier
        xgb_eval_metric = "mlogloss"
    else:
        gbm_class = GradientBoostingRegressor
        ada_class = AdaBoostRegressor
        xgb_class = XGBRegressor
        lgb_class = LGBMRegressor
        cat_class = CatBoostRegressor
        xgb_eval_metric = "rmse"

    model_configs = {}

    # GradientBoost (scikit-learn)
    model_configs["GradientBoost_sklearn"] = {
        "model_class": gbm_class,
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": random_state
        }
    }

    # AdaBoost (scikit-learn)
    model_configs["AdaBoost_sklearn"] = {
        "model_class": ada_class,
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "random_state": random_state
        }
    }

    # XGBoost
    if xgb is not None and xgb_class is not None:
        model_configs["XGBoost"] = {
            "model_class": xgb_class,
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": random_state,
                "use_label_encoder": False,  # для новых версий XGBoost
                "eval_metric": xgb_eval_metric
            }
        }

    # LightGBM
    if lgb is not None and lgb_class is not None:
        model_configs["LightGBM"] = {
            "model_class": lgb_class,
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": random_state
            }
        }

    # CatBoost
    if cat_class is not None:
        model_configs["CatBoost"] = {
            "model_class": cat_class,
            "params": {
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 3,
                "random_state": random_state,
                "silent": True  # отключаем детальные логи CatBoost
            }
        }

    return model_configs


def get_memory_usage_mb():
    """ Возвращает текущий объем использования памяти в МБ для данного процесса. """
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)
