import numpy as np
import copy
from sklearn.tree import DecisionTreeClassifier

class ModestAdaBoostClassifier:
    def __init__(self, estimator=None, n_estimators=500, learning_rate=1, random_state=None):
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.learning_rate = learning_rate
        self.estimators = []
        self.probs_pos = []
        self.probs_pos_inv = []
        self.probs_neg = []
        self.probs_neg_inv = []
        self.random_state = random_state
        self.eps=0#1e-6

    def fit(self, X, y):
        self.estimators = []
        n_samples = X.shape[0]
        D_t = np.ones(n_samples) / n_samples
        for t in range(self.n_estimators):

            h_t = copy.deepcopy(self.estimator)
            h_t.fit(X, y, sample_weight=D_t)
            self.estimators.append(h_t)

            D_inv_t = 1 - D_t
            D_inv_t = D_inv_t / D_inv_t.sum()
            
            if type(self.estimator) == DecisionTreeClassifier and h_t.get_depth() == 1:
                pred_leaves = h_t.apply(X)

                probs_pos_t = np.empty(h_t.get_n_leaves())
                probs_pos_inv_t = np.empty(h_t.get_n_leaves())
                probs_neg_t = np.empty(h_t.get_n_leaves())
                probs_neg_inv_t = np.empty(h_t.get_n_leaves())
                
                for leaf_ind in range(1, h_t.get_n_leaves()+1):
                    probs_pos_t[leaf_ind-1] = np.where((pred_leaves == leaf_ind) *  (h_t.predict(X)==1) * (y==1), D_t, 0).sum() / np.where(leaf_ind == pred_leaves, D_t, 0).sum()
                    probs_pos_inv_t[leaf_ind-1] = np.where((pred_leaves == leaf_ind) *  (h_t.predict(X)==1) * (y==1), D_inv_t, 0).sum() / np.where(leaf_ind == pred_leaves, D_inv_t, 0).sum()
                    probs_neg_t[leaf_ind-1] = np.where((pred_leaves == leaf_ind) *  (h_t.predict(X)==0) * (y==0), D_t, 0).sum() / np.where(leaf_ind == pred_leaves, D_t, 0).sum()
                    probs_neg_inv_t[leaf_ind-1] = np.where((pred_leaves == leaf_ind) *  (h_t.predict(X)==0) * (y==0), D_inv_t, 0).sum() / np.where(leaf_ind == pred_leaves, D_inv_t, 0).sum()

                self.probs_pos.append(probs_pos_t)
                self.probs_pos_inv.append(probs_pos_inv_t)
                self.probs_neg.append(probs_neg_t)
                self.probs_neg_inv.append(probs_neg_inv_t)
            else:
                raise NotImplementedError

            pred_probs_pos_t = np.clip(np.take_along_axis(probs_pos_t, pred_leaves-1, 0), self.eps, 1 - self.eps) 
            pred_probs_pos_inv_t = np.clip(np.take_along_axis(probs_pos_inv_t, pred_leaves-1, 0), self.eps, 1 - self.eps) 
            pred_probs_neg_t = np.clip(np.take_along_axis(probs_neg_t, pred_leaves-1, 0), self.eps, 1 - self.eps) 
            pred_probs_neg_inv_t = np.clip(np.take_along_axis(probs_neg_inv_t, pred_leaves-1, 0), self.eps, 1 - self.eps) 

            pred = pred_probs_pos_t * (1-pred_probs_pos_inv_t) - pred_probs_neg_t * (1-pred_probs_neg_inv_t)
            D_t *= np.exp(-pred * self.learning_rate * np.where(y>0, 1, -1))
            D_t /= D_t.sum()
        return self

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for t in range(self.n_estimators):

            pred_leaves = self.estimators[t].apply(X)

            probs_pos_t = self.probs_pos[t]
            probs_pos_inv_t = self.probs_pos_inv[t]
            probs_neg_t = self.probs_neg[t]
            probs_neg_inv_t = self.probs_neg_inv[t]

            pred_probs_pos_t = np.clip(np.take_along_axis(probs_pos_t, pred_leaves-1, 0), self.eps, 1 - self.eps) 
            pred_probs_pos_inv_t = np.clip(np.take_along_axis(probs_pos_inv_t, pred_leaves-1, 0), self.eps, 1 - self.eps) 
            pred_probs_neg_t = np.clip(np.take_along_axis(probs_neg_t, pred_leaves-1, 0), self.eps, 1 - self.eps) 
            pred_probs_neg_inv_t = np.clip(np.take_along_axis(probs_neg_inv_t, pred_leaves-1, 0), self.eps, 1 - self.eps) 

            pred = pred_probs_pos_t * (1-pred_probs_pos_inv_t) - pred_probs_neg_t * (1-pred_probs_neg_inv_t)
            y += pred
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