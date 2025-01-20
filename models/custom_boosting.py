import numpy as np

class MadaBoostClassifier:
    def __init__(self, n_estimators=50, random_state=42, base_estimator=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.base_estimator = base_estimator
        self.ensemble_ = []
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        from sklearn.tree import DecisionTreeClassifier
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=1)

        for i in range(self.n_estimators):
            clf = self._clone_estimator()
            bootstrap_inds = np.random.randint(0, len(X), len(X))
            X_boot = X.iloc[bootstrap_inds]
            y_boot = y.iloc[bootstrap_inds]
            clf.fit(X_boot, y_boot)
            self.ensemble_.append(clf)
        return self

    def predict(self, X):
        preds = []
        for clf in self.ensemble_:
            preds.append(clf.predict(X))
        preds = np.array(preds).T
        final = []
        for row in preds:
            vals, counts = np.unique(row, return_counts=True)
            final.append(vals[np.argmax(counts)])
        return np.array(final)

    def _clone_estimator(self):
        import copy
        return copy.deepcopy(self.base_estimator)


class BrownBoostClassifier:
    def __init__(self, n_estimators=50, random_state=42, base_estimator=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.base_estimator = base_estimator
        self.ensemble_ = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        from sklearn.tree import DecisionTreeClassifier
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=1)

        for i in range(self.n_estimators):
            clf = self._clone_estimator()
            bootstrap_inds = np.random.randint(0, len(X), len(X))
            X_boot = X.iloc[bootstrap_inds]
            y_boot = y.iloc[bootstrap_inds]
            clf.fit(X_boot, y_boot)
            self.ensemble_.append(clf)
        return self

    def predict(self, X):
        preds = []
        for clf in self.ensemble_:
            preds.append(clf.predict(X))
        preds = np.array(preds).T
        final = []
        for row in preds:
            vals, counts = np.unique(row, return_counts=True)
            final.append(vals[np.argmax(counts)])
        return np.array(final)

    def _clone_estimator(self):
        import copy
        return copy.deepcopy(self.base_estimator)


class FilterBoostClassifier:
    def __init__(self, n_estimators=50, random_state=42, base_estimator=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.base_estimator = base_estimator
        self.ensemble_ = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        from sklearn.tree import DecisionTreeClassifier
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=1)
        indices = np.arange(len(X))
        for i in range(self.n_estimators):
            clf = self._clone_estimator()
            if i == 0:
                selected_inds = indices
            else:
                y_pred = self.ensemble_[-1].predict(X.iloc[selected_inds])
                errors = (y_pred != y.iloc[selected_inds])
                selected_inds = selected_inds[errors]

                if len(selected_inds) == 0:
                    break

            if len(selected_inds) > 0:
                X_sel = X.iloc[selected_inds]
                y_sel = y.iloc[selected_inds]
                clf.fit(X_sel, y_sel)
                self.ensemble_.append(clf)
            else:
                break

        return self

    def predict(self, X):
        if not self.ensemble_:
            return np.random.choice(np.unique(X), size=len(X))

        preds = []
        for clf in self.ensemble_:
            preds.append(clf.predict(X))
        preds = np.array(preds).T
        final = []
        for row in preds:
            vals, counts = np.unique(row, return_counts=True)
            final.append(vals[np.argmax(counts)])
        return np.array(final)

    def _clone_estimator(self):
        import copy
        return copy.deepcopy(self.base_estimator)
