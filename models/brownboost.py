# Courtesy of https://github.com/lapis-zero09/BrownBoost


import math
import numpy as np
from scipy.special import erf
import copy

try:
    import cupy as cp
    from cupyx.scipy.special import erf as cp_erf
    from cuml.tree import DecisionTreeClassifier as GPUDecisionTree
except ImportError:
    GPUDecisionTree = None


class BrownBoostClassifier:
    def __init__(self, estimator=None, c=4, convergence_criterion=0.0001, n_estimators=20_000, max_iter_newton_raphson=10000, random_state=None):
        """ Initiates BrownBoost classifier

        Parameters
        ----------
        estimator: classifier from scikit-learn
            The base leaner in ensemble
        c: int or float
            A positive real value
            default = 10
        convergence_criterion: float
            A small constant(>0) used to avoid degenerate cases.
            default = 0.0001
        """
        self.estimator = estimator
        self.c = c
        self.n_estimators = n_estimators
        self.max_iter_newton_raphson = max_iter_newton_raphson
        self.convergence_criterion = convergence_criterion
        self.alphas = []
        self.estimators = []
        self.random_state = random_state

    def fit(self, X, y):
        """ Trains the classifier
        Parameters
        ----------
        X: ndarray
            The training instances
        y: ndarray
            The target values for The training instances

        returns
        --------
            self
        """
        self.alphas = []
        self.estimators = []
        # Initiate parameters
        s = self.c
        r = np.zeros(X.shape[0])
        k = 0
        while s >= 0 and k < self.n_estimators:
            #             print(f'iter is {k}\ts = {s}')
            k += 1
            w = np.exp(-(r + s)**2 / self.c)
            if w.sum()==0:
                return self
            h = copy.deepcopy(self.estimator)
            h.fit(X, y, sample_weight=w)
            pred = h.predict(X)

            error = np.where(pred == y, 1., -1.)
            gamma = np.dot(w, error)

            alpha, t = self.newton_raphson(r, error, s, gamma)


#             theta = (0.1/self.c)**2
#             A = 32 * math.sqrt(self.c*math.log(2/theta))
#             if t < gamma**2/A:
#                 (new_t * w).sum()
#                 t = new_t + gamma**2/A

            r += alpha * error
            s -= t
            self.alphas.append(alpha)
            self.estimators.append(h)
        return self

    def predict(self, X):
        """ Classify the samples
        Parameters
        ----------
        X: ndarray
            The test instances

        Returns
        -------
        y: ndarray
            The pred with BrownBoost for the test instances
        """
        y = np.zeros(X.shape[0])
        for i in range(0, len(self.estimators)):
            y += self.alphas[i] * self.estimators[i].predict(X)
        return (y / sum(self.alphas) > 0.5)

    def newton_raphson(self, r, error, s, gamma):
        """ Computes alpha and t
        Parameters
        ----------
        r: array
            margins for the instances
        error: ndarray
            error vec between pred and true instances
        s: float
            'time remaining'
        gamma: float
            correlation
        y: ndarray
            the target values

        Retruns
        -------
        alpha: float
        t: float
        """

        # Theorem 3 & 5
        alpha = min([0.1, gamma])
        t = (alpha**2) / 3

        a = r + s
        change_amount = self.convergence_criterion + 1
        k = 0
        error += 1e-10

        while change_amount > self.convergence_criterion and k < self.max_iter_newton_raphson and t >= s and alpha>t > self.convergence_criterion:
            d = a + alpha * error - t
            w = np.exp(-d**2 / self.c)

            # Coefficients for jacobian
            W = w.sum()
            U = (w * d * error).sum()
            B = (w * error).sum()
#             if abs(B) < 0.001:
#                 break
            V = (w * d * error**2).sum()
            E = (erf(d / math.sqrt(self.c)) - erf(a / math.sqrt(self.c))).sum()

            sqrt_pi_c = math.sqrt(math.pi * self.c)
            denominator = 2*(V*W - U*B) + 1e-10

            alpha_step = (self.c*W*B + sqrt_pi_c*U*E)/denominator
            t_step = (self.c*B*B + sqrt_pi_c*V*E)/denominator

            alpha += alpha_step
            t += t_step

            change_amount = math.sqrt(alpha_step**2 + t_step**2)
#             print(f'\t newton_raphson iter is {k}, {change_amount}')
            k += 1

        return alpha, t

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "n_estimators": self.n_estimators,
            "c": self.c,
            "convergence_criterion": self.convergence_criterion,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.c = params.get("c", self.c)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.convergence_criterion = params.get(
            "convergence_criterion", self.convergence_criterion)
        self.random_state = params.get("random_state", self.random_state)
        return self


class BrownBoostClassifierGPU:
    def __init__(self, estimator=None, c=4, convergence_criterion=0.001, n_estimators=200000, random_state=None):
        self.estimator = estimator
        self.c = c
        self.n_estimators = n_estimators
        self.max_iter_newton_raphson = 100
        self.convergence_criterion = convergence_criterion
        self.alphas = []
        self.estimators = []
        self.random_state = random_state

    def fit(self, X, y):
        xg = cp.asarray(X)
        yg = cp.asarray(y)
        self.alphas = []
        self.estimators = []
        s = self.c
        r = cp.zeros(xg.shape[0], dtype=cp.float32)
        k = 0

        while s >= 0 and k < self.n_estimators:
            k += 1
            w = cp.exp(-(r + s)**2 / self.c)

            if GPUDecisionTree and isinstance(self.estimator, GPUDecisionTree):
                h = copy.deepcopy(self.estimator)
                h.fit(xg, yg, sample_weight=w)
                p = h.predict(xg)
                p = (p > 0.5).astype(cp.float32)
            elif GPUDecisionTree and isinstance(self.estimator, BrownBoostClassifierGPU):
                h = GPUDecisionTree(max_depth=1)
                h.fit(xg, yg, sample_weight=w)
                p = h.predict(xg)
                p = (p > 0.5).astype(cp.float32)
            else:
                h = copy.deepcopy(self.estimator)
                h.fit(cp.asnumpy(xg), cp.asnumpy(yg),
                      sample_weight=cp.asnumpy(w))
                p = cp.asarray(h.predict(cp.asnumpy(xg)))

            e = cp.where(p == yg, 1., -1.)
            g = cp.dot(w, e)
            a, t = self.newton_raphson(r, e, s, g)
            r += a*e
            s -= t
            self.alphas.append(a)
            self.estimators.append(h)

        return self

    def predict(self, X):
        xg = cp.asarray(X)
        s = 0
        y = cp.zeros(xg.shape[0], dtype=cp.float32)

        for i in range(len(self.estimators)):
            if GPUDecisionTree and isinstance(self.estimators[i], GPUDecisionTree):
                p = self.estimators[i].predict(xg)
                p = (p > 0.5).astype(cp.float32)
            else:
                p = cp.asarray(self.estimators[i].predict(cp.asnumpy(xg)))

            y += self.alphas[i]*p
            s += self.alphas[i]

        return (y > 0.5*s).get()

    def newton_raphson(self, r, e, s, g):
        a = cp.minimum(0.1, g)
        t = (a**2)/3
        A = r+s
        c = self.convergence_criterion+1
        k = 0
        e = e+1e-10

        while c > self.convergence_criterion and k < self.max_iter_newton_raphson:
            d = A+a*e-t
            w = cp.exp(-d**2/self.c)
            W = w.sum()
            U = (w*d*e).sum()
            B = (w*e).sum()
            V = (w*d*(e**2)).sum()

            try:
                E = cp.sum(cp_erf(d/cp.sqrt(self.c))-cp_erf(A/cp.sqrt(self.c)))
            except:
                dcp = d.get()
                Acp = A.get()
                E = cp.asarray((erf(dcp / math.sqrt(self.c)) -
                               erf(Acp / math.sqrt(self.c))).sum())

            D = 2*(V*W - U*B)+1e-10
            as_ = (self.c*W*B+math.sqrt(math.pi*self.c)*U*E)/D
            ts_ = (self.c*B*B+math.sqrt(math.pi*self.c)*V*E)/D
            a += as_
            t += ts_
            c = cp.sqrt(as_**2+ts_**2)
            k += 1

        return float(a.get()), float(t.get())

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def get_params(self, deep=True):
        return {"convergence_criterion": self.convergence_criterion, "c": self.c, "n_estimators": self.n_estimators, "estimator": self.estimator, "random_state": self.random_state}

    def set_params(self, **p):
        self.estimator = p.get("estimator", self.estimator)
        self.c = p.get("c", self.c)
        self.n_estimators = p.get("n_estimators", self.n_estimators)
        self.convergence_criterion = p.get(
            "convergence_criterion", self.convergence_criterion)
        self.random_state = p.get("random_state", self.random_state)

        return self


def get_brownboost_class(gpu=False):
    return BrownBoostClassifierGPU if gpu else BrownBoostClassifier
