import numpy as np
from copy import deepcopy
from ngboost import NGBClassifier

class NGBoostClassifier:
    def __init__(self, estimator=None, *, n_estimators=500,
                 learning_rate=0.01, natural_gradient=True,
                 verbose=False, random_state=None):
        self.base = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.natural_gradient = natural_gradient
        self.verbose = verbose
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None


    def _encode_labels(self, y):
        self.classes_, y_int = np.unique(y, return_inverse=True)
        return y_int.astype(int)

    def _decode_labels(self, y_int):
        return self.classes_.take(y_int, axis=0)


    def fit(self, X, y):
        X = np.asarray(X)
        y_int = self._encode_labels(np.asarray(y))

        self.model_ = NGBClassifier(
            Base=deepcopy(self.base),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            natural_gradient=self.natural_gradient,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        self.model_.fit(X, y_int)
        return self

    def predict(self, X):
        y_int = self.model_.predict(X)
        return self._decode_labels(y_int)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


    def get_params(self, deep=True):
        return {
            "estimator": self.base,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "natural_gradient": self.natural_gradient,
            "verbose": self.verbose,
            "random_state": self.random_state,
        }


    def set_params(self, **p):
        self.base = p.get("estimator", self.base)
        self.n_estimators = p.get("n_estimators", self.n_estimators)
        self.learning_rate = p.get("learning_rate", self.learning_rate)
        self.natural_gradient = p.get("natural_gradient", self.natural_gradient)
        self.verbose = p.get("verbose", self.verbose)
        self.random_state = p.get("random_state", self.random_state)
        return self


def get_ngboost_class():
    return NGBoostClassifier
