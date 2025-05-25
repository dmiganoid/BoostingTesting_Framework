import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from copy import deepcopy



class MWaterBoostClassifier:
    def __init__(self, estimator=None, n_estimators=20, learning_rate=1):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        self.weights = None
        self.weight_modifiers = None

    def fit(self, X, y):
        self.estimator_weights = []
        self.estimators = []
        self.betas = []
        n_samples = X.shape[0]

        D_t = np.ones(n_samples) / n_samples
        B_t = np.ones(n_samples)
        for t in range(self.n_estimators):

            h_t = deepcopy(self.estimator)
            h_t.fit(X, y, sample_weight=D_t)
            pred = h_t.predict(X)

            err_t = np.sum(D_t * (pred != y))
            alpha_t = self.learning_rate * 0.5 * np.log((1-err_t +1e-10)/(err_t+1e-10))

            self.estimator_weights.append(alpha_t)
            self.estimators.append(h_t)

            
            
            B_t *= np.where(pred==y, np.exp(-alpha_t), np.exp(alpha_t))
            D_t = np.where( B_t <= 1,
                           D_t * B_t, # D_0 * B_t
                           1/n_samples)
            if (1-(D_t).sum()) > 0:
                decreased_weight = 1-(D_t).sum() #prev - now
                increased_weight_d = np.where(B_t > 1, B_t, 0)
                if increased_weight_d.sum() > 0:
                    D_t += decreased_weight * increased_weight_d / increased_weight_d.sum()
            if D_t.sum()==0:
                return self
            D_t /= D_t.sum() #if no weights to increase or sum of D_t > W  

        return self

    def predict(self, X, sign=True):
        y = np.zeros(X.shape[0])
        for i in range(len(self.estimators)):
            y += self.estimator_weights[i]*self.estimators[i].predict(X)

        if sign:
            return y/ sum(self.estimator_weights) >= 0.5
        return y / sum(self.estimator_weights)

    def score(self, X, y):
        return (self.predict(X, sign=True)==y).sum()/X.shape[0]
    
    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "estimator": self.estimator, "learning_rate": self.learning_rate, "random_state": self.random_state}

    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.random_state = params.get("random_state", self.random_state)
        return self


    
class XWaterBoostClassifier:
    def __init__(self, estimator=None, n_estimators=20, learning_rate=1):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        self.weights = None
        self.weight_modifiers = None

    def fit(self, X, y):
        self.estimator_weights = []
        self.estimators = []
        self.betas = []
        n_samples = X.shape[0]

        D_t = np.ones(n_samples) / n_samples
        B_t = np.ones(n_samples)
        for t in range(self.n_estimators):

            h_t = deepcopy(self.estimator)
            h_t.fit(X, y, sample_weight=D_t)
            pred = h_t.predict(X)

            err_t = np.sum(D_t * (pred != y)) + 1e-10
            alpha_t = self.learning_rate * 0.5 * np.log((1-err_t)/err_t)

            self.estimator_weights.append(alpha_t)
            self.estimators.append(h_t)

            
            
            B_t *= np.where(pred==y, np.exp(-alpha_t), np.exp(alpha_t))
            D_t = np.where( B_t <= 1,
                           D_t * B_t, # D_0 * B_t
                           1/n_samples)
            if (1-(D_t).sum()):
                decreased_weight = 1-(D_t).sum() #prev - now
                increased_weight_d = np.where(B_t > 1, B_t, 0)
                if increased_weight_d.sum() > 0:
                    D_t += decreased_weight * increased_weight_d / increased_weight_d.sum()
            if D_t.sum()==0:
                self.n_estimators = t
                return self
            D_t /= D_t.sum() #if no weights to increase or sum of D_t > W  

        return self

    def predict(self, X, sign=True):
        y = np.zeros(X.shape[0])
        for i in range(len(self.estimators)):
            y += self.estimator_weights[i]*self.estimators[i].predict(X)

        if sign:
            return y/ sum(self.estimator_weights) >= 0.5
        return y / sum(self.estimator_weights)

    def score(self, X, y):
        return (self.predict(X, sign=True)==y).sum()/X.shape[0]
    
    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "estimator": self.estimator, "learning_rate": self.learning_rate, "random_state": self.random_state}

    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.random_state = params.get("random_state", self.random_state)
        return self
