import numpy as np
import copy

try:
    import cupy as cpx
    from cuml.tree import DecisionTreeClassifier as GPUDecisionTree
except:
    GPUDecisionTree = None


class SOWAdaBoostClassifier:
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.alphas = []
        self.estimators = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        D_t = np.ones(n_samples) / n_samples

        self.alphas = []
        self.estimators = []
        
        for _ in range(self.n_estimators):
            h = copy.deepcopy(self.estimator)
            h.fit(X, y, sample_weight=D_t)
            
            pred = h.predict(X)
            
            miss = (pred != y)
            err_t = np.sum(D_t * miss)
            err_t = min(max(err_t, 1e-16), 1 - 1e-16) # divizion by zero protection
            
            alpha_t = 0.5 * np.log((1 - err_t) / err_t) * self.learning_rate
            self.alphas.append(alpha_t)
            self.estimators.append(h)
            

            margin = np.where(miss, -1.0, 1.0)
            
            grad = -np.exp(-margin)
            hess = np.exp(-margin)
            
            factor = np.exp(self.learning_rate * np.abs(grad) * hess)
            
            D_t = D_t * factor
            
            D_t /= np.sum(D_t)

        return self

    def predict(self, X, sign=True):
        n_samples = X.shape[0]
        agg = np.zeros(n_samples)
        sum_alpha = 0
        for alpha_t, h in zip(self.alphas, self.estimators):
            p = h.predict(X)
            agg += alpha_t * p
            sum_alpha += alpha_t
        if sign:
            return (agg > 0.5 * sum_alpha).astype(int)
        else:
            return agg / sum_alpha

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


class SOWAdaBoostClassifierGPU:
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.alphas = []
        self.estimators = []

    def fit(self, X, y):
        if cpx is None:
            raise ImportError("cupy is not installed or not available.")
        xg = cpx.asarray(X) 
        yg = cpx.asarray(y)
        n = xg.shape[0]
        D_t = cpx.ones(n, dtype=cpx.float32) / n

        self.alphas = []
        self.estimators = []

        for _ in range(self.n_estimators):
            if GPUDecisionTree and isinstance(self.estimator, GPUDecisionTree):
                h = copy.deepcopy(self.estimator)
                h.fit(xg, yg, sample_weight=D_t)
                p = h.predict(xg).astype(cpx.float32)
            else:
                h = copy.deepcopy(self.estimator)
                h.fit(cpx.asnumpy(xg), cpx.asnumpy(yg), sample_weight=cpx.asnumpy(D_t))
                p = cpx.asarray(h.predict(cpx.asnumpy(xg)), dtype=cpx.float32)

            miss = (p != yg)
            err = (D_t * miss).sum()
            err = cpx.clip(err, 1e-10, 1 - 1e-10)

            alpha_t = 0.5 * cpx.log((1 - err) / err) * self.learning_rate
            self.alphas.append(float(alpha_t.get()))
            self.estimators.append(h)

            margin = cpx.where(miss, -1.0, 1.0)

            grad = -cpx.exp(-margin)
            hess = cpx.exp(-margin)

            factor = cpx.exp(self.learning_rate * cpx.abs(grad) * hess)

            D_t = D_t * factor
            D_t = D_t / D_t.sum()

        return self

    def predict(self, X, sign=True):
        xg = cpx.asarray(X, dtype=cpx.float32)
        agg = cpx.zeros(xg.shape[0], dtype=cpx.float32)
        sum_alpha = 0
        for i, h in enumerate(self.estimators):
            if GPUDecisionTree and isinstance(h, GPUDecisionTree):
                p = h.predict(xg).astype(cpx.float32)
            else:
                p = cpx.asarray(h.predict(cpx.asnumpy(xg)), dtype=cpx.float32)
            agg += self.alphas[i] * p
            sum_alpha += self.alphas[i]
        if sign:
            return (agg > 0.5 * sum_alpha).get().astype(int)
        else:
            return (agg / sum_alpha).get()

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def get_params(self):
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


def get_sowadaboost_class(gpu=False):
    return SOWAdaBoostClassifierGPU if gpu else SOWAdaBoostClassifier
