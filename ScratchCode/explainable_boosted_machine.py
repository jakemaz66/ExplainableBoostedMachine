from ScratchCode.boosting_algorithm import GradientBoostingRegressor
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


class EBM:

    def __init__(self, n_iterations, learning_rate=0.01, max_depth=3):

        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.stored_trees = []


    def train(self, features_x: pd.DataFrame, target_y: pd.Series):
        #Initialize predictions with the mean of the target values
        y_pred = pd.Series(target_y.mean(), index=target_y.index)

        for _ in range(self.n_iterations):

            for feature in features_x.columns:
                #Calculate residuals
                residuals = target_y - y_pred

                #Fit the decision tree on the current feature and residuals
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                #Sklearn DecisionTreeRegressor Expects Numeric Inputs
                tree.fit(features_x[[feature]], residuals)

                #Predict the residuals using the fitted tree
                pred = tree.predict(features_x[[feature]])

                #Update predictions with the learning rate and predicted residuals
                y_pred += self.learning_rate * pred

                #Store the trained tree
                self.stored_trees.append((feature, tree))


    def predict(self, features_x: pd.DataFrame):
        """This method predicts y_pred using individual features

        Args:
            features_x (pd.DataFrame): The dataframe of features

        Returns:
            dict: A dictionary where keys are individual features and values are predictions only using the feature
        """
        #Initialize predictions with the mean of the target values
        feature_pred = {}
        mean = pd.Series(0, index=features_x.index) + self.stored_trees[0][1].predict(features_x[[self.stored_trees[0][0]]])

        # dd contributions from all stored trees
        for feature in features_x.columns:

            y_pred = mean

            for trained_feature, tree in self.stored_trees:

                if trained_feature == feature:

                    y_pred += self.learning_rate * tree.predict(features_x[[feature]])

            feature_pred[feature] = y_pred 

        return feature_pred
    

    def plot_feature(self, features_x: pd.DataFrame, feature_name: str,  target_y: pd.Series):
        """
        Plot the predictions of y for each threshold value of the specified feature.

        Args:
            features_x (pd.DataFrame): The dataframe of features.
            feature_name (str): The name of the feature to plot.

        Returns:
            None
        """
         #Filter tree list to only include trees for the feature of interest
        feature_trees = [tree for y, tree in self.stored_trees if y == feature_name]
        plot = {}
        
        #Get the feature values and create a range of values for the feature to test
        feature = features_x[feature_name]
        min_val, max_val = feature.min(), feature.max()
        
        #Get 100 values between the minimum and maximum feature value to plot
        feature_values = np.linspace(min_val, max_val, num=100)
        
        #Initialize the predictions with the mean prediction
        mean = target_y.mean()
        
        for value in feature_values:
            #Update predictions for the current value of the feature
            y_pred = mean
            
            #Create a copy of the DataFrame and set the feature to the current value
            temp_features_x = features_x.copy()
            temp_features_x[feature_name] = value
            
            for tree in feature_trees:
                y_pred += self.learning_rate * tree.predict(temp_features_x[[feature_name]])
            
            plot[value] = y_pred.mean()  

        #Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(list(plot.keys()), list(plot.values()), marker='o')
        plt.xlabel(feature_name)
        plt.ylabel('Predicted y')
        plt.title(f'Prediction of y vs {feature_name}')
        plt.grid(True)
        plt.show()

        

if __name__ == '__main__':

    df = pd.read_csv(
    r"C:\Users\jakem\Downloads\archive (19)\insurance.csv")

    #Predicting charges from all the other features
    X = df.iloc[:, :-2]
   
    sex_mapper = {
       'male': 0,
       'female': 1
   }
    
    smoke_mapper = {
        'yes': 0,
        'no': 1
    }
    
    X['sex'] = X['sex'].map(sex_mapper)
    X['smoker'] = X['smoker'].map(smoke_mapper)

    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 

    ebm = EBM(5000)
    ebm.train(X_train, y_train)

    #Obtaining the error for predicting y_test using each individual feature
    predictions = ebm.predict(X_test)
    errors = {}

    for feature in predictions.keys():
        error = mean_squared_error(predictions[feature], y_test)
        errors[feature] = error 

    print(errors)



