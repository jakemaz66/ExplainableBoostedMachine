import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        #The number of trees
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.initial_pred = None

    def fit(self, X, y):
        #Step 1: Initialize predictions with the mean of the target values
        self.initial_pred = np.mean(y)
        y_pred = np.full(y.shape, self.initial_pred)

        for _ in range(self.n_estimators):
            #Step 2: Calculate residuals (negative gradient)
            residuals = y - y_pred

            #Step 3: Train a decision tree on the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            #Step 4: Update predictions
            y_pred += self.learning_rate * tree.predict(X)

            #Store the trained tree so I can get the output of all of them at inference
            self.models.append(tree)

    def predict(self, X):
        #Start with the initial prediction
        y_pred = np.full((X.shape[0],), self.initial_pred)

        #Add the contribution of each tree
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred
    

if __name__ == '__main__':
    pass