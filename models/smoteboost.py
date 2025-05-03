import numpy as np
from copy import deepcopy
from imblearn.over_sampling import SMOTE

class SMOTEBoostClassifier:
    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0,
                 sampling_strategy="auto", random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        self.alphas = []
        self.estimators_ = []
        self.classes_ = None
        self._smote = SMOTE(sampling_strategy=self.sampling_strategy,
                            random_state=self.random_state)


    def _check_binary(self, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Only binaty classification currently supported.")
        self._y2signed = {self.classes_[0]: -1, self.classes_[1]: 1}


    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self._check_binary(y)

        y_signed = np.vectorize(self._y2signed.get)(y)
        n_samples = X.shape[0]

        D_t = np.ones(n_samples, dtype=float) / n_samples

        rng = np.random.default_rng(self.random_state)
        self.alphas = []
        self.estimators_ = []

        for t in range(self.n_estimators):
            X_res, y_res = self._smote.fit_resample(X, y)

            sample_weight_res = np.zeros(X_res.shape[0])
            sample_weight_res[:n_samples] = D_t              
            if X_res.shape[0] > n_samples:
                minority_mask = (y == self.classes_[1])
                minority_mean_w = D_t[minority_mask].mean() if minority_mask.any() else D_t.mean()
                sample_weight_res[n_samples:] = minority_mean_w

            h_t = deepcopy(self.estimator)
            h_t.fit(X_res, y_res, sample_weight=sample_weight_res)

            pred = h_t.predict(X)
            pred_signed = np.vectorize(self._y2signed.get)(pred)
            miss = (pred != y)
            err_t = np.dot(D_t, miss)
            err_t = np.clip(err_t, 1e-16, 1 - 1e-16)

            alpha_t = self.learning_rate * 0.5 * np.log((1 - err_t) / err_t)

            margin = y_signed * pred_signed
            D_t *= np.exp(-alpha_t * margin)
            D_t /= D_t.sum()

            self.alphas.append(alpha_t)
            self.estimators_.append(h_t)

        return self

    def _raw_margin(self, X):
        agg = np.zeros(X.shape[0])
        for alpha, h in zip(self.alphas, self.estimators_):
            pred = h.predict(X)
            agg += alpha * np.vectorize(self._y2signed.get)(pred)
        return agg

    def predict(self, X):
        margins = self._raw_margin(X)
        return np.where(margins >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "sampling_strategy": self.sampling_strategy,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.sampling_strategy = params.get("sampling_strategy", self.sampling_strategy)
        self.random_state = params.get("random_state", self.random_state)
        self._smote = SMOTE(sampling_strategy=self.sampling_strategy,
                            random_state=self.random_state)
        return self
    
def get_smoteboost_class():
    return SMOTEBoostClassifier
