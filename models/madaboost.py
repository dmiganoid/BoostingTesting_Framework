import numpy as np
import copy

try:
    import cupy as cpx
    from cuml.tree import DecisionTreeClassifier as GPUDecisionTree
except:
    GPUDecisionTree = None



class MadaBoostClassifier:
    def __init__(self, estimator=None, n_estimators=500, learning_rate=1, random_state=None):
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.learning_rate = learning_rate
        self.alphas = []
        self.estimators = []
        self.random_state = random_state

    def fit(self, X, y):
        self.alphas = []
        self.estimators = []
        self.betas = []
        n_samples = X.shape[0]

        D_0 = np.ones(n_samples) / n_samples
        B_t = np.ones(n_samples)
        D_t = D_0
        for t in range(self.n_estimators):

            h_t = copy.deepcopy(self.estimator)
            h_t.fit(X, y, sample_weight=D_t)
            pred = h_t.predict(X)

            err_t = np.sum(D_t * (pred != y))
            alpha_t = self.learning_rate * 0.5 * np.log((1-err_t +1e-10)/(err_t+1e-10))
            beta_t = np.exp(-alpha_t)

            self.alphas.append(alpha_t)
            self.estimators.append(h_t)

            B_t *= np.where(pred==y, beta_t, 1/beta_t)
            D_t = np.where( B_t <= 1, D_0*B_t, D_0)
            if D_t.sum()==0:
                self.n_estimators = t
                return self
            D_t /= D_t.sum()
        return self

    def predict(self, X, sign=True):
        y = np.zeros(X.shape[0])
        for t in range(self.n_estimators):
            y += self.alphas[t] * self.estimators[t].predict(X)
        if sign:
            return (y / sum(self.alphas) > 0.5)
        else:
            return y / sum(self.alphas)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.random_state = params.get("random_state", self.random_state)

        return self


class MadaBoostClassifierGPU:
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.alphas = []
        self.estimators = []

    def fit(self, X, y):
        xg = cpx.asarray(X) if cpx is not None else None
        yg = cpx.asarray(y) if cpx is not None else None
        n = xg.shape[0]
        D_t = cpx.ones(n, dtype=cpx.float32) / n
        self.alphas = []
        self.estimators = []
        for _ in range(self.n_estimators):
            if GPUDecisionTree and isinstance(self.estimator, GPUDecisionTree):
                h = copy.deepcopy(self.estimator)
                h.fit(xg, yg, sample_weight=D_t)
                p = h.predict(xg)
                p = (p > 0.5).astype(cpx.float32)
            elif GPUDecisionTree and isinstance(self.estimator, MadaBoostClassifierGPU):
                h = GPUDecisionTree(max_depth=1)
                h.fit(xg, yg, sample_weight=D_t)
                p = h.predict(xg)
                p = (p > 0.5).astype(cpx.float32)
            else:
                h = copy.deepcopy(self.estimator)
                h.fit(cpx.asnumpy(xg), cpx.asnumpy(yg),
                      sample_weight=cpx.asnumpy(D_t))
                p = cpx.asarray(h.predict(cpx.asnumpy(xg)))
            e = (p != yg)
            err = (D_t * e).sum() - 1e-10
            a = 0.5 * cpx.log((1 - err) / err) * self.learning_rate
            self.alphas.append(float(a.get()))
            self.estimators.append(h)
            fac = cpx.where(e, cpx.exp(a), cpx.exp(-a))
            D_t = D_t * fac
            D_t = D_t / D_t.sum()
        return self

    def predict(self, X, sign=True):
        xg = cpx.asarray(X) if cpx is not None else None
        agg = cpx.zeros(xg.shape[0], dtype=cpx.float32)
        s = 0
        for i, h in enumerate(self.estimators):
            if GPUDecisionTree and isinstance(h, GPUDecisionTree):
                p = h.predict(xg)
                p = (p > 0.5).astype(cpx.float32)
            else:
                p = cpx.asarray(h.predict(cpx.asnumpy(xg)))
            agg += self.alphas[i] * p
            s += self.alphas[i]
        if sign:
            return (agg > 0.5 * s).get()
        else:
            return (agg / s).get()

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "estimator": self.estimator, "learning_rate": self.learning_rate, "random_state": self.random_state}

    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.random_state = params.get("random_state", self.random_state)
        return self


def get_madaboost_class(gpu=False):
    return MadaBoostClassifierGPU if gpu else MadaBoostClassifier
