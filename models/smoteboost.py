import numpy as np
from copy import deepcopy
from imblearn.over_sampling import SMOTE

class SMOTEBoostClassifier:
    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0,
                 sampling_strategy="auto", k_neighbors=5,
                 random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

        self.classes_ = None
        self.alphas_ = []
        self.estimators_ = []

    def _check_inputs(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, y_int = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("SMOTEBoost supports only binary classification")
        return X, y_int.astype(int)

    def _update_distribution(self, D, y_true, y_pred, alpha):
        miss = (y_pred != y_true)
        D *= np.exp(self.learning_rate * alpha * miss)

        if D.sum() == 0:
            return self

        D /= D.sum()
        
        return D

    def fit(self, X, y):
        X, y_int = self._check_inputs(X, y)
        rng = np.random.default_rng(self.random_state)

        n_samples = X.shape[0]
        D = np.ones(n_samples, dtype=float) / n_samples

        self.alphas_ = []
        self.estimators_ = []

        minority_mask = (y_int == 1)

        for _ in range(self.n_estimators):
            idx = rng.choice(n_samples, size=n_samples, replace=True, p=D)
            X_res, y_res = X[idx], y_int[idx]

            seed = rng.integers(0, 2**32)
            smote = SMOTE(
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
                random_state=int(seed)
            )

            X_syn, y_syn = smote.fit_resample(X_res, y_res)

            n_orig = X_res.shape[0]
            n_syn = X_syn.shape[0] - n_orig

            sample_weight = np.zeros(X_syn.shape[0], dtype=float)
            sample_weight[:n_orig] = D[idx]

            if n_syn > 0:
                avg_minority_w = D[minority_mask].mean()
                sample_weight[n_orig:] = avg_minority_w

            h = deepcopy(self.estimator)
            h.fit(X_syn, y_syn, sample_weight=sample_weight)

            y_pred = h.predict(X)
            err = np.dot(D, (y_pred != y_int))
            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)

            D = self._update_distribution(D, y_int, y_pred, alpha)

            self.alphas_.append(alpha)
            self.estimators_.append(h)

        return self

    def _aggregate(self, X):
        agg = np.zeros(X.shape[0], dtype=float)
        for alpha, h in zip(self.alphas_, self.estimators_):
            pred = h.predict(X)
            pred_signed = np.where(pred == 1, 1, -1)
            agg += alpha * pred_signed
        return agg

    def predict(self, X):
        margin = self._aggregate(np.asarray(X))
        y_pred_int = (margin >= 0).astype(int)
        return self.classes_[y_pred_int]

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "sampling_strategy": self.sampling_strategy,
            "k_neighbors": self.k_neighbors,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
