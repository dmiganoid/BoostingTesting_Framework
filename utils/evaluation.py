# gbm_benchmark/utils/evaluation.py

"""
Модуль для дополнительных метрик и функций, если понадобится.
Можно добавить F1, ROC-AUC, RMSE, MAE и т.д.
"""


def rmse(y_true, y_pred):
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    return sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)
