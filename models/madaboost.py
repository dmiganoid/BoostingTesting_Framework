import numpy as np
import copy as cp



class MadaBoost:
    def __init__(self, weaklearner, n_estimators=500):
        self.n_estimators = n_estimators
        self.weaklearner = weaklearner
        self.alphas = []
        self.models = []


    def fit(self, X, y):
        n_samples = X.shape[0]
        D_t = np.ones(n_samples) / n_samples
        for t in range(self.n_estimators):

            h_t = cp.deepcopy(self.weaklearner)
            h_t.fit(X, y, sample_weight=D_t)
            pred = h_t.predict(X)

            err_t = np.sum(D_t * (pred != y)) + 1e-10

            beta_t = np.sqrt(err_t/(1-err_t))
            alpha_t = np.log(1/beta_t)
            
            self.alphas.append(alpha_t)
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