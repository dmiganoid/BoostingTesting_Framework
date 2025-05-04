import numpy as np
from copy import deepcopy
from ngboost import NGBClassifier

class NGBoostClassifier:
    def __init__(self, estimator=None, *, n_estimators=500,
                 learning_rate=0.01, natural_gradient=True,
                 verbose=False, random_state=None,
                 retry_on_singular=True):
        self.base = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.natural_gradient = natural_gradient
        self.verbose = verbose
        self.random_state = random_state
        self.retry_on_singular = retry_on_singular

        self.model_ = None
        self.classes_ = None

    def _encode_labels(self, y):
        self.classes_, y_int = np.unique(y, return_inverse=True)
        return y_int.astype(int)

    def _decode_labels(self, y_int):
        return self.classes_.take(y_int, axis=0)

    def _make_ngb(self):
        return NGBClassifier(
            Base=deepcopy(self.base),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            natural_gradient=self.natural_gradient,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        import warnings, numpy as np
        X = np.asarray(X)
        y_int = self._encode_labels(np.asarray(y))

        self.model_ = self._make_ngb()
        try:
            self.model_.fit(X, y_int)
        except np.linalg.LinAlgError as e:
            if not self.retry_on_singular:
                raise
            warnings.warn(
                "NGBoost encountered singular matrix; retrying with natural_gradient=False",
                RuntimeWarning,
            )

            self.natural_gradient = False
            self.model_ = self._make_ngb()
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
            "retry_on_singular": self.retry_on_singular,
        }

    def set_params(self, **p):
        self.base = p.get("estimator", self.base)
        self.n_estimators = p.get("n_estimators", self.n_estimators)
        self.learning_rate = p.get("learning_rate", self.learning_rate)
        self.natural_gradient = p.get("natural_gradient", self.natural_gradient)
        self.verbose = p.get("verbose", self.verbose)
        self.random_state = p.get("random_state", self.random_state)
        self.retry_on_singular= p.get("retry_on_singular", self.retry_on_singular)
        return self
