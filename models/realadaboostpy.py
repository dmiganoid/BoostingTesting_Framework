import numpy as np
import copy


class RealAdaBoostClassifier:
    def __init__(self, estimator=None, n_estimators=500, learning_rate=1, random_state=None):
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.learning_rate = learning_rate
        self.estimators = []
        self.random_state = random_state
        self.eps=1e-6

    def fit(self, X, y):
        self.estimators = []
        n_samples = X.shape[0]

        D_t = np.ones(n_samples) / n_samples
        for t in range(self.n_estimators):

            h_t = copy.deepcopy(self.estimator)
            h_t.fit(X, y, sample_weight=D_t)
            
            pred = h_t.predict_proba(X)[:, 1]
            p = np.clip(pred, self.eps, 1 - self.eps)
            q = np.clip(1-pred, self.eps, 1 - self.eps)
            pred = 1/2 *np.log(p/q)
            
            self.estimators.append(h_t)

            D_t *= np.exp(-pred * self.learning_rate * np.where(y>0, 1, -1))
            D_t /= D_t.sum()
        return self

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for t in range(self.n_estimators):
            pred = self.estimators[t].predict_proba(X)[:, 1]
            p = np.clip(pred, self.eps, 1 - self.eps)
            q = np.clip(1-pred, self.eps, 1 - self.eps)
            y += 1/2 *np.log(p/q)

        return np.where(y > 0, self.estimators[0].classes_[1], self.estimators[0].classes_[0])

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "estimator": self.estimator,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.random_state = params.get("random_state", self.random_state)

        return self