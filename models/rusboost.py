from copy import deepcopy
import numpy as np
from sklearn.utils import check_random_state
from collections import Counter


class RUSBoostClassifier:
    def __init__(self, estimator=None, *,
                 n_estimators=50, learning_rate=1.0,
                 sampling_strategy="auto", random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.estimators_ = []
        self.alphas_ = []
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        rng = check_random_state(self.random_state)
        X = np.asarray(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Только бинарная классификация поддерживается")


        counts = Counter(y)
        min_label, maj_label = sorted(counts, key=lambda lbl: counts[lbl])
        y_pm = np.where(y == self.classes_[1], 1, -1)


        if min_label != self.classes_[1]:
            y_pm = -y_pm  

        n_samples = X.shape[0]
        if sample_weight is None:
            D = np.full(n_samples, 1/n_samples, dtype=float)
        else:
            D = sample_weight / np.sum(sample_weight)

        self.estimators_, self.alphas_ = [], []

        for m in range(self.n_estimators):
            min_mask = (y == min_label)
            maj_mask = ~min_mask
            idx_min = np.where(min_mask)[0]
            idx_maj = np.where(maj_mask)[0]
            n_min, n_maj = len(idx_min), len(idx_maj)
            if n_min == 0 or n_maj == 0:
                break

            if self.sampling_strategy == "auto":
                k = n_min
            else:
                k = int(self.sampling_strategy * n_min)

            p = D[idx_maj] / D[idx_maj].sum()
            sel_maj = rng.choice(idx_maj, size=min(k, n_maj), replace=False, p=p)
            sel_idx = np.concatenate([idx_min, sel_maj])
            X_bal, y_bal = X[sel_idx], y[sel_idx]
            D_bal = D[sel_idx]
            D_bal /= D_bal.sum()

            h = deepcopy(self.estimator)
            h.fit(X_bal, y_bal, sample_weight=D_bal)

            pred = h.predict(X)
            pred_pm = np.where(pred == min_label, 1, -1)
            if min_label != self.classes_[1]:
                pred_pm = -pred_pm
            err = np.sum(D * (pred_pm != y_pm))

            if err >= 0.5:
                continue
            alpha = (0.5 * self.learning_rate *
                     np.log((1 - err) / err)) if err > 0 else 1.0

            self.estimators_.append(h)
            self.alphas_.append(alpha)
            D *= np.exp(-alpha * y_pm * pred_pm)
            D /= D.sum()

        return self


    def _aggregate(self, X):
        agg = np.zeros(X.shape[0])
        for a, h in zip(self.alphas_, self.estimators_):
            pred = h.predict(X)
            agg += a * np.where(pred == self.classes_[1], 1, -1)
        return agg

    def predict(self, X):
        scores = self._aggregate(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        scores = self._aggregate(X)
        prob_pos = 1 / (1 + np.exp(-2 * scores / np.sum(np.abs(self.alphas_))))
        prob_neg = 1 - prob_pos
        return np.vstack([prob_neg, prob_pos]).T

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

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self
