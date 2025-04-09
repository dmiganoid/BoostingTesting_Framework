import numpy as np
import copy
from sklearn.tree import DecisionTreeRegressor

try:
    import cupy as cpx
    from cuml.tree import DecisionTreeRegressor as GPUDecisionTreeRegressor
except:
    GPUDecisionTreeRegressor = None

def _init_constant_logloss(y):
    p = np.mean(y)
    eps = 1e-5
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def _negative_gradient_logloss(y, y_pred):
    prob = 1.0 / (1.0 + np.exp(-y_pred))
    return y - prob

def _line_search_logloss(y, y_pred, base_pred):
    return 1.0

def _init_constant_exp(y):
    return 0.0

def _negative_gradient_exp(y, y_pred):
    y_ = 2*y - 1
    return y_ * np.exp(-y_ * y_pred)

def _line_search_exp(y, y_pred, base_pred):
    return 1.0

LOSS_FUNCTIONS_CPU = {
    "log_loss": (_init_constant_logloss, _negative_gradient_logloss, _line_search_logloss),
    "exponential": (_init_constant_exp, _negative_gradient_exp, _line_search_exp),
}

class GradientBoostingClassifier:
    def __init__(self, estimator=None, n_estimators=100, learning_rate=0.1, loss="log_loss", random_state=None):
        self.estimator = estimator if estimator is not None else DecisionTreeRegressor()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.models_ = []
        self.gammas_ = []
        self.F0_ = 0.0
        if self.loss not in LOSS_FUNCTIONS_CPU:
            raise ValueError(f"Unknown loss {self.loss}")
        self._init_func, self._neg_grad_func, self._line_search_func = LOSS_FUNCTIONS_CPU[self.loss]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.F0_ = self._init_func(y)
        F = np.full(X.shape[0], self.F0_, dtype=float)
        self.models_ = []
        self.gammas_ = []
        for _ in range(self.n_estimators):
            residuals = self._neg_grad_func(y, F)
            base = copy.deepcopy(self.estimator)
            base.fit(X, residuals)
            base_pred = base.predict(X)
            gamma_m = self._line_search_func(y, F, base_pred)
            F += self.learning_rate * gamma_m * base_pred
            self.models_.append(base)
            self.gammas_.append(gamma_m)
        return self

    def _predict_raw(self, X):
        X = np.asarray(X, dtype=float)
        raw_pred = np.full(X.shape[0], self.F0_, dtype=float)
        for base, gamma_m in zip(self.models_, self.gammas_):
            raw_pred += self.learning_rate * gamma_m * base.predict(X)
        return raw_pred

    def predict(self, X):
        raw_pred = self._predict_raw(X)
        if self.loss == "log_loss":
            prob = 1.0 / (1.0 + np.exp(-raw_pred))
            return (prob > 0.5).astype(np.float32)
        elif self.loss == "exponential":
            return (raw_pred > 0.0).astype(np.float32)
        else:
            return (raw_pred > 0.5).astype(np.float32)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "estimator": self.estimator,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.loss = params.get("loss", self.loss)
        self.random_state = params.get("random_state", self.random_state)
        if self.loss not in LOSS_FUNCTIONS_CPU:
            raise ValueError(f"Unknown loss: {self.loss}")
        self._init_func, self._neg_grad_func, self._line_search_func = LOSS_FUNCTIONS_CPU[self.loss]
        return self

def _init_constant_logloss_gpu(y):
    p = cpx.mean(y)
    eps = 1e-5
    p = cpx.clip(p, eps, 1 - eps)
    return cpx.log(p / (1 - p))

def _negative_gradient_logloss_gpu(y, y_pred):
    prob = 1.0 / (1.0 + cpx.exp(-y_pred))
    return y - prob

def _line_search_logloss_gpu(y, y_pred, base_pred):
    return 1.0

def _init_constant_exp_gpu(y):
    return 0.0

def _negative_gradient_exp_gpu(y, y_pred):
    y_ = 2*y - 1
    return y_ * cpx.exp(-y_ * y_pred)

def _line_search_exp_gpu(y, y_pred, base_pred):
    return 1.0

LOSS_FUNCTIONS_GPU = {
    "log_loss": (_init_constant_logloss_gpu, _negative_gradient_logloss_gpu, _line_search_logloss_gpu),
    "exponential": (_init_constant_exp_gpu, _negative_gradient_exp_gpu, _line_search_exp_gpu),
}

class GradientBoostingClassifierGPU:
    def __init__(self, estimator=None, n_estimators=100, learning_rate=0.1, loss="log_loss", random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.models_ = []
        self.gammas_ = []
        self.F0_ = 0.0
        if self.loss not in LOSS_FUNCTIONS_GPU:
            raise ValueError(f"Unknown loss: {self.loss}")
        self._init_func, self._neg_grad_func, self._line_search_func = LOSS_FUNCTIONS_GPU[self.loss]

    def fit(self, X, y):
        xg = cpx.asarray(X, dtype=cpx.float32)
        yg = cpx.asarray(y, dtype=cpx.float32)
        self.F0_ = self._init_func(yg)
        Fg = cpx.full(xg.shape[0], self.F0_, dtype=cpx.float32)
        self.models_ = []
        self.gammas_ = []
        for _ in range(self.n_estimators):
            residuals = self._neg_grad_func(yg, Fg)
            if GPUDecisionTreeRegressor and isinstance(self.estimator, GPUDecisionTreeRegressor):
                base = copy.deepcopy(self.estimator)
                base.fit(xg, residuals)
                base_pred = base.predict(xg)
            else:
                base = copy.deepcopy(self.estimator) if self.estimator else None
                if base is None:
                    from sklearn.tree import DecisionTreeRegressor
                    base = DecisionTreeRegressor()
                base.fit(cpx.asnumpy(xg), cpx.asnumpy(residuals))
                base_pred = cpx.asarray(base.predict(cpx.asnumpy(xg)), dtype=cpx.float32)
            gamma_m = self._line_search_func(yg, Fg, base_pred)
            Fg = Fg + self.learning_rate * gamma_m * base_pred
            self.models_.append(base)
            self.gammas_.append(gamma_m)
        return self

    def _predict_raw(self, X):
        xg = cpx.asarray(X, dtype=cpx.float32)
        raw_pred = cpx.full(xg.shape[0], self.F0_, dtype=cpx.float32)
        for base, gamma_m in zip(self.models_, self.gammas_):
            if GPUDecisionTreeRegressor and isinstance(base, GPUDecisionTreeRegressor):
                p = base.predict(xg)
            else:
                p = cpx.asarray(base.predict(cpx.asnumpy(xg)), dtype=cpx.float32)
            raw_pred += self.learning_rate * gamma_m * p
        return raw_pred

    def predict(self, X):
        raw_pred = self._predict_raw(X)
        if self.loss == "log_loss":
            prob = 1.0 / (1.0 + cpx.exp(-raw_pred))
            return (prob > 0.5).astype(cpx.float32).get()
        elif self.loss == "exponential":
            return (raw_pred > 0.0).astype(cpx.float32).get()
        else:
            return (raw_pred > 0.5).astype(cpx.float32).get()

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean()

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "estimator": self.estimator,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.loss = params.get("loss", self.loss)
        self.random_state = params.get("random_state", self.random_state)
        if self.loss not in LOSS_FUNCTIONS_GPU:
            raise ValueError(f"Unknown loss: {self.loss}")
        self._init_func, self._neg_grad_func, self._line_search_func = LOSS_FUNCTIONS_GPU[self.loss]
        return self

def get_gradientboost_class(gpu=False):
    return GradientBoostingClassifierGPU if gpu else GradientBoostingClassifier
