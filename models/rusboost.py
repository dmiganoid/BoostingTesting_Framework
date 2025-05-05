from copy import deepcopy
import numpy as np
from sklearn.utils import check_random_state


class RUSBoostClassifier:
    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=50,
        learning_rate=1.0,
        sampling_strategy="auto",
        random_state=None,
    ):
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

        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("RUSBoostClassifier supports only binary classes")

        y_pm = np.where(y == self.classes_[1], 1, -1)

        n_samples = X.shape[0]
        if sample_weight is None:
            D = np.full(n_samples, 1 / n_samples, dtype=float)
        else:
            D = sample_weight / np.sum(sample_weight)

        self.estimators_ = []
        self.alphas_ = []

        for m in range(self.n_estimators):
            min_mask = y_pm == 1
            maj_mask = ~min_mask

            idx_min = np.where(min_mask)[0]
            idx_maj = np.where(maj_mask)[0]

            n_min = idx_min.shape[0]
            if n_min == 0 or idx_maj.shape[0] == 0:
                break

            if idx_maj.shape[0] <= n_min:
                sel_maj = idx_maj
            else:
                p_maj = D[idx_maj] / D[idx_maj].sum()
                sel_maj = rng.choice(idx_maj, size=n_min, replace=False, p=p_maj)

            sel_idx = np.concatenate([idx_min, sel_maj])
            X_bal, y_bal = X[sel_idx], y[sel_idx]

            D_bal = D[sel_idx]
            D_bal /= D_bal.sum()

            h = deepcopy(self.estimator)
            h.fit(X_bal, y_bal, sample_weight=D_bal)

            pred_pm = np.where(h.predict(X) == self.classes_[1], 1, -1)
            err = np.sum(D * (pred_pm != y_pm))

            if err >= 0.5 - 1e-10:
                break
            if err == 0:
                alpha = 1.0
            else:
                alpha = self.learning_rate * 0.5 * np.log((1 - err) / err)

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
