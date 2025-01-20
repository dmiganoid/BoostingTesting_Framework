# gbm_benchmark/data_loader.py

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


def generate_synthetic_data(
    task_type='classification',
    n_samples=10000,
    n_features=20,
    random_state=42
):
    """
    Генерирует синтетический датасет для классификации или регрессии.
    Возвращает (X, y) как DataFrame и Series соответственно.
    """
    np.random.seed(random_state)
    if task_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_redundant=2,
            random_state=random_state
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            noise=0.1,
            random_state=random_state
        )

    return pd.DataFrame(X), pd.Series(y)


def load_data_from_csv(path):
    """
    Пример функции для чтения данных из CSV.
    Предполагается, что в CSV последняя колонка - это целевая переменная (y).
    Или вы можете адаптировать под себя.
    """
    df = pd.read_csv(path)
    # допустим, предположим, что последняя колонка - это target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y
