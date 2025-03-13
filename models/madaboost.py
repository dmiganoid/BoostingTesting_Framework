import numpy as np
import copy as cp



class MadaBoost:
    def __init__(self, estimator=None, n_estimators=500, learning_rate=1):
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.learning_rate = learning_rate
        self.alphas = []
        self.models = []


    def fit(self, X, y):
        self.alphas = []
        self.hs = []
        n_samples = X.shape[0]
        D_t = np.ones(n_samples) / n_samples
        for t in range(self.n_estimators):

            h_t = cp.deepcopy(self.estimator)
            h_t.fit(X, y, sample_weight=D_t)
            pred = h_t.predict(X)

            err_t = np.sum(D_t * (pred != y)) + 1e-10

            beta_t = np.sqrt(err_t/(1-err_t))
            alpha_t = np.log(1/beta_t)
            
            self.alphas.append(alpha_t*self.learning_rate)
            self.models.append(h_t)


            D_t = np.where(D_t*beta_t**(pred*y)<=1/n_samples, D_t*beta_t**(pred*y), 1/n_samples)
            D_t /= D_t.sum()

    def predict(self, X, sign=True):
        prediction = np.zeros(X.shape[0])
        for t in range(self.n_estimators):
            prediction += self.alphas[t] * self.models[t].predict(X)
        if sign:
            return np.sign(prediction).astype(int)
        else:
            return prediction
        
    def score(self, X, y):
        return (self.predict(X)==y).sum()/X.shape[0]
    
    def get_params(self, deep=True):
        return {
            "n_estimators" : self.n_estimators,
            "estimator" : self.estimator,
            "learning_rate" : self.learning_rate
             }
    
    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        return self