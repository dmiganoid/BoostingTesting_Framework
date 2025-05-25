from copy import deepcopy
import numpy as np
from sklearn.utils import check_random_state
from collections import Counter


class RUSBoostClassifier:
    """
    RUSBoost for binary classification with random undersampling of the majority class.
    """

    def __init__(self, estimator=None, *,
                 n_estimators=50,
                 learning_rate=1.0,
                 sampling_strategy="auto",
                 random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        self.classes_ = None
        self.min_label_ = None
        self.maj_label_ = None
        self.estimators_ = []
        self.alphas_ = []

    def fit(self, X, y, sample_weight=None):
        rng = check_random_state(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("Only binary classification is supported")

        counts = Counter(y)
        self.min_label_, self.maj_label_ = sorted(counts, key=lambda lbl: counts[lbl])
        y_pm = np.where(y == self.min_label_, 1, -1)

        n_samples = X.shape[0]
        if sample_weight is None:
            D = np.full(n_samples, 1.0 / n_samples, dtype=float)
        else:
            D = sample_weight.astype(float)
            if D.sum() == 0:
                return self
            D /= D.sum()

        self.estimators_.clear()
        self.alphas_.clear()

        for m in range(self.n_estimators):
            idx_min = np.where(y == self.min_label_)[0]
            idx_maj = np.where(y == self.maj_label_)[0]

            if idx_min.size == 0 or idx_maj.size == 0:
                break

            if self.sampling_strategy == "auto":
                k = idx_min.size
            else:
                k = int(self.sampling_strategy * idx_min.size)

            p_maj = D[idx_maj] / D[idx_maj].sum()
            sel_maj = rng.choice(idx_maj, size=min(k, idx_maj.size), replace=False, p=p_maj)

            sel_idx = np.concatenate([idx_min, sel_maj])
            X_bal, y_bal = X[sel_idx], y[sel_idx]
            D_bal = D[sel_idx].copy()
            D_bal /= D_bal.sum()

            h = deepcopy(self.estimator)
            h.fit(X_bal, y_bal, sample_weight=D_bal)

            pred = h.predict(X)
            pred_pm = np.where(pred == self.min_label_, 1, -1)

            err = np.sum(D * (pred_pm != y_pm))

            if err == 0:
                alpha = 1e6
            else:
                alpha = 0.5 * self.learning_rate * np.log((1 - err) / err)

            self.estimators_.append(h)
            self.alphas_.append(alpha)

            D *= np.exp(-alpha * y_pm * pred_pm)
            if D.sum() == 0:
                return self
            D /= D.sum()

        return self

    def _aggregate(self, X):
        X = np.asarray(X)
        agg = np.zeros(X.shape[0], dtype=float)
        for alpha, h in zip(self.alphas_, self.estimators_):
            pred = h.predict(X)
            pred_pm = np.where(pred == self.min_label_, 1, -1)
            agg += alpha * pred_pm
        return agg

    def predict(self, X):
        scores = self._aggregate(X)
        return np.where(scores >= 0, self.min_label_, self.maj_label_)

    def predict_proba(self, X):
        scores = self._aggregate(X)

        prob_min = 1.0 / (1.0 + np.exp(-2.0 * scores))
        prob_maj = 1.0 - prob_min

        if self.classes_[0] == self.min_label_:
            return np.vstack([prob_min, prob_maj]).T
        else:
            return np.vstack([prob_maj, prob_min]).T

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "sampling_strategy": self.sampling_strategy,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
