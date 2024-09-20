import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns


def example():
    """This function runs a test example of the interpret show on a toy dataset
    """

    df = pd.read_csv(
        r"C:\Users\jakem\Downloads\adult\adult.data")

    df.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]

    X = df.iloc[:, :-1]
    y = (df.iloc[:, -1] == " >50K").astype(int)

    #Encoding the nominal variables for the logistic regression
    pd.concat([X, pd.get_dummies(X['WorkClass'])])
    X.drop('WorkClass', axis=1, inplace=True)

    pd.concat([X, pd.get_dummies(X['Education'])])
    X.drop('Education', axis=1, inplace=True)

    pd.concat([X, pd.get_dummies(X['MaritalStatus'])])
    X.drop('MaritalStatus', axis=1, inplace=True)

    pd.concat([X, pd.get_dummies(X['Race'])])
    X.drop('Race', axis=1, inplace=True)

    pd.concat([X, pd.get_dummies(X['Gender'])])
    X.drop('Gender', axis=1, inplace=True)

    pd.concat([X, pd.get_dummies(X['NativeCountry'])])
    X.drop('NativeCountry', axis=1, inplace=True)

    pd.concat([X, pd.get_dummies(X['Occupation'])])
    X.drop('Occupation', axis=1, inplace=True)

    pd.concat([X, pd.get_dummies(X['Relationship'])])
    X.drop('Relationship', axis=1, inplace=True)

    seed = 42
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    #Feature importance only available for linear kernel type
    def f_importances(coef, names):
        imp = coef
        imp,names = zip(*sorted(zip(imp,names)))
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        plt.show()

    f_importances(svm_classifier.coef_, X_train.columns)

if __name__ == '__main__':
    example()