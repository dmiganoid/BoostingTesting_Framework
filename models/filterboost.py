import numpy as np
import copy as cp
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_hastie_10_2, make_classification
#from sklearn.tree import DecisionTreeClassifier
class Oracle:
    def __init__(self, X, y, random_state):
        self.X = X
        self.y = y
        self.rng = np.random.default_rng(seed=random_state)
    def __call__(self):
        index = self.rng.integers(0, self.X.shape[0])
        return self.X[index], self.y[index]
    
    
class NaiveFilterBoostClassifier:
    def __init__(self, estimator=None, n_estimators=100, epsilon=0.1, delta=0.9, tau=0.1, learning_rate=1, m_t=None, random_state=None):
        """
        Initialize FilterBoost.
        estimator: Number of iterations
        epsilon (0,1): Target error
        delta (0,1): Confidence parameter
        tau (0,1): Relative edge error
        """
        self.WL = estimator
        self.n_estimators = n_estimators
        self.epsilon = epsilon
        self.delta = delta
        self.estimators = []  # List of (alpha, weak_learner) pairs
        self.alphas = []
        self.r = 0
        self.m_t = m_t
        self.oracle = None
        self.delta_t = None
        self.t = 0
        self.tau = tau
        self.converged = True
        self.learning_rate = learning_rate
        self.random_state = random_state

    def compute_F(self, X):
        """Compute F_t(x) for current hypothesis."""
        y = np.zeros(X.shape[0])
        for i in range(len(self.estimators)):
            y += self.alphas[i] * self.estimators[i].predict(X)
        return y

    def Filter(self):
        """
        Filter function to get (x, y) with probability q_t(x, y).
        Returns (x, y) and previous H (simplified as F).
        """
        self.r += 1
        # delta_t_cor = self.delta_t / (self.r * (self.r + 1))
        for _ in range(1000000): #for i in range(int(np.ceil(2/self.epsilon*np.log(1 / delta_t_cor)))):
            x, y = self.oracle()
            q_t = 1 / (1 + np.exp(y * (self.compute_F(x.reshape(1, -1))[0] if self.estimators else 0.0)))
            if np.random.random() < q_t:  # Sample with probability q_t
                return x, y

    def getEdge(self, new_estimator):
        """
        Compute edge γ_t using sampling from Filter.
        """
        m, n, u, alpha = 0, 0, 0, np.inf
        #if len(self.estimators)==0:

        while abs(u) < alpha * (1 + 1 / self.tau) and n < self.oracle.X.shape[0]:
            x, y = self.Filter()
            n += 1
            m += new_estimator.predict(x.reshape(1, -1)) == y  # I(h_t(x) = y)
            u = m / n - 0.5
            alpha = np.sqrt(0.5/n * np.log((n * (n + 1)) / self.delta_t))
        if n == self.oracle.X.shape[0]:
            self.converged = False
        return u / (1 + self.tau)


    def fit(self, X, y):
        """Main FilterBoost algorithm."""
        self.alphas = []
        self.estimators = []
        if self.m_t is None:
            self.m_t = lambda t: X.shape[0] / 4
        
        self.oracle = Oracle(X, np.where(y, 1, -1), self.random_state)
        self.oracle = Oracle(X, y, self.random_state)
        
        for t in range(1, self.n_estimators+1):
            self.r = 0
            self.delta_t = self.delta/(3*t*(t+1)) 
            #self.delta_t = self.delta / (3*t*(t + 1))
            x_t, y_t = [], []
            for _ in range(int(self.m_t(t))):
                filtered_object = self.Filter()
                x_t.append(filtered_object[0])
                y_t.append(filtered_object[1])
            
            # Train weak learner
            h_t = cp.deepcopy(self.WL)
            h_t.fit(x_t, y_t)
            
            # Compute edge
            gamma_t = self.getEdge(h_t)

            # Compute alpha
            alpha_t = 0.5 * np.log((1/2 + gamma_t) / (1/2 - gamma_t - 1e-10))

            # Update F
            self.estimators.append(h_t)
            self.alphas.append(alpha_t*self.learning_rate)


    def predict(self, X, sign=True):
        y = np.zeros(X.shape[0])
        for t in range(self.n_estimators):
            y += self.alphas[t] * self.estimators[t].predict(X)
        if sign:
            return np.sign(y).astype(int)
        else:
            return y
        
    def score(self, X, y):
        return (self.predict(X)==y).mean()
    
    def get_params(self, deep=True):
        return {
            "estimator" : self.estimator,
            "n_estimators" : self.n_estimators,
            "learning_rate" : self.learning_rate,
            "epsilon" : self.epsilon, 
            "delta" : self.delta, 
            "tau" : self.tau,
            "random_state" : self.random_state
             }
    
    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.epsilon = params.get("epsilon", self.epsilon)
        self.delta = params.get("delta", self.delta)
        self.tau = params.get("tau", self.tau)
        self.random_state = params.get("random_state", self.random_state)
        return self


class FilterBoostClassifier:
    def __init__(self, estimator=None, n_estimators=100, epsilon=0.1, delta=0.9, tau=0.1, learning_rate=1, m_t=None, random_state=None):
        """
        Initialize FilterBoost.
        estimator: Number of iterations
        epsilon (0,1): Target error
        delta (0,1): Confidence parameter
        tau (0,1): Relative edge error
        """
        self.WL = estimator
        self.n_estimators = n_estimators
        self.epsilon = epsilon
        self.delta = delta
        self.estimators = []  # List of (alpha, weak_learner) pairs
        self.alphas = []
        self.r = 0
        self.m_t = m_t
        self.oracle = None
        self.delta_t = None
        self.t = 0
        self.tau = tau
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)
        
    def fit(self, X, y):
        """Main FilterBoost algorithm."""
        self.alphas = []
        self.estimators = []
        for t in range(1, self.n_estimators+1):
            self.delta_t = self.delta/(3*t*(t+1))
            self.r = 0
            if len(self.estimators) > 0:
                ind = np.where(self.rng.random(size=X.shape[0]) <= 1 / (1 + np.exp(np.where(y == self.predict(X, sign=False), 1, -1))))
            else:
                ind = np.where(self.rng.random(size=X.shape[0]) >= 0)
            self.r += X.shape[0]
             

            # Train weak learner
            h_t = cp.deepcopy(self.WL)
            h_t.fit(X[ind], y[ind])

            
            # Compute edge
            m, n, u, alpha = 0, 0, 0, np.inf
            inds = np.where(self.rng.random(size=X.shape[0]) <= 1 / (1 + np.exp(np.where(y == self.predict(X, sign=False), 1, -1))))
            self.rng.shuffle(inds)
            errs = h_t.predict(X[ind]) == y[ind]
            for err in errs:
                self.r += 1
                #delta_t_cor = self.delta_t / (self.r * (self.r + 1)) -> filter fails i=int(np.ceil(2/self.epsilon*np.log(1 / delta_t_cor))) times means end of fitting
                if abs(u) < alpha * (1 + 1 / self.tau):
                    n += 1
                    m += err  # I(h_t(x) = y)
                    u = m / n - 0.5
                    alpha_t = np.sqrt(0.5/n * np.log((n * (n + 1)) / self.delta_t))
                else:
                    break
            gamma_t = u / (1 + self.tau)

            # Compute alpha
            alpha_t = 0.5 * np.log((1/2 + gamma_t) / (1/2 - gamma_t))

            # Update F
            self.estimators.append(h_t)
            self.alphas.append(alpha_t*self.learning_rate)
        
        return self


    def predict(self, X, sign=True):
        y = np.zeros(X.shape[0])
        for t in range(len(self.estimators)):
            y += self.alphas[t] * self.estimators[t].predict(X)
        if sign:
            return (y/sum(self.alphas) >=0.5).astype(int)
        else:
            return y
        
    def score(self, X, y):
        return (self.predict(X)==y).mean()
    
    def get_params(self, deep=True):
        return {
            "estimator" : self.estimator,
            "n_estimators" : self.n_estimators,
            "learning_rate" : self.learning_rate,
            "epsilon" : self.epsilon, 
            "delta" : self.delta, 
            "tau" : self.tau,
            "random_state" : self.random_state
             }
    
    def set_params(self, **params):
        self.estimator = params.get("estimator", self.estimator)
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.epsilon = params.get("epsilon", self.epsilon)
        self.delta = params.get("delta", self.delta)
        self.tau = params.get("tau", self.tau)
        self.random_state = params.get("random_state", self.random_state)
        self.rng = np.random.default_rng(seed=self.random_state)
        return self