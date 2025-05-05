import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor

class NGBoostClassifier:
    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=500,
        learning_rate=0.01,
        natural_gradient=True,
        minibatch_frac=1.0,
        verbose=False,
        random_state=None,
    ):
        self.base_estimator = estimator if estimator is not None else DecisionTreeRegressor(max_depth=3)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.natural_gradient = natural_gradient
        self.minibatch_frac = minibatch_frac
        self.verbose = verbose
        self.random_state = random_state

        self.F0_ = 0.0
        self.models_ = []
        self.scales_ = []
        self.classes_ = None
        self._rng = np.random.default_rng(self.random_state)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _nll(y, f):
        p = NGBoostClassifier._sigmoid(f)
        eps = 1e-12
        return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()

    def _sample(self, X, y, f):
        if self.minibatch_frac >= 1.0:
            return X, y, f
        m = int(self.minibatch_frac * X.shape[0])
        idx = self._rng.choice(X.shape[0], m, replace=False)
        return X[idx], y[idx], f[idx]

    def _grad(self, y, f):
        p = self._sigmoid(f)
        g = p - y
        if self.natural_gradient:
            fisher = p * (1 - p)
            g = g / (fisher + 1e-12)
        return g

    def _line_search(self, y, f, direction):
        base_loss = self._nll(y, f)
        scale = 1.0
        while scale > 1e-4:
            new_f = f - self.learning_rate * scale * direction
            loss = self._nll(y, new_f)
            if loss < base_loss:
                return scale
            scale *= 0.5
        return 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes, y_int = np.unique(y, return_inverse=True)
        if classes.size != 2:
            raise ValueError("Only binary classification is supported.")
        self.classes_ = classes
        y = y_int.astype(float)

        p0 = np.clip(y.mean(), 1e-5, 1 - 1e-5)
        self.F0_ = np.log(p0 / (1 - p0))
        f = np.full(X.shape[0], self.F0_, dtype=float)
        self.models_, self.scales_ = [], []

        for m_idx in range(self.n_estimators):
            X_batch, y_batch, f_batch = self._sample(X, y, f)
            g = self._grad(y_batch, f_batch)
            base = deepcopy(self.base_estimator)
            base.fit(X_batch, g)
            direction = base.predict(X)
            scale = self._line_search(y, f, direction)
            if scale == 0.0:
                if self.verbose:
                    print(f"Stopped at iter {m_idx}: line‑search failed")
                break
            f -= self.learning_rate * scale * direction
            self.models_.append(base)
            self.scales_.append(scale)
            if self.verbose and (m_idx+1) % 50 == 0:
                print(f"iter {m_idx+1}  NLL={self._nll(y, f):.4f}  scale={scale:.4f}")
        return self

    def _raw_decision(self, X):
        X = np.asarray(X, dtype=float)
        f = np.full(X.shape[0], self.F0_, dtype=float)
        for base, s in zip(self.models_, self.scales_):
            f -= self.learning_rate * s * base.predict(X)
        return f

    def predict_proba(self, X):
        raw = self._raw_decision(X)
        prob = self._sigmoid(raw)
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = (proba[:, 1] >= 0.5).astype(int)
        return self.classes_.take(idx, axis=0)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def get_params(self, deep=True):
        return {
            "estimator": self.base_estimator,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "natural_gradient": self.natural_gradient,
            "minibatch_frac": self.minibatch_frac,
            "verbose": self.verbose,
            "random_state": self.random_state,
        }

    def set_params(self, **p):
        self.base_estimator = p.get("estimator", self.base_estimator)
        self.n_estimators = p.get("n_estimators", self.n_estimators)
        self.learning_rate = p.get("learning_rate", self.learning_rate)
        self.natural_gradient = p.get("natural_gradient", self.natural_gradient)
        self.minibatch_frac = p.get("minibatch_frac", self.minibatch_frac)
        self.verbose = p.get("verbose", self.verbose)
        self.random_state = p.get("random_state", self.random_state)
        self._rng = np.random.default_rng(self.random_state)
        return self