import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


def generate_synthetic_data(
    task_type='classification',
    n_samples=10000,
    n_features=20,
    random_state=42
):
    np.random.seed(random_state)
    if task_type == 'classification':
        n_informative = min(n_features, 10)
        n_redundant = min(max(0, n_features - n_informative), 2)

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
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
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y
