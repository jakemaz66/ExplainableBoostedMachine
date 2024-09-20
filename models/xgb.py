import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import DMatrix, plot_importance


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

    #Changing categorical type from 'Object' to 'Categorical'
    categorical_columns = ["WorkClass", "Education", "MaritalStatus", "Occupation",
                           "Relationship", "Race", "Gender", "NativeCountry"]
    for col in categorical_columns:
        X[col] = X[col].astype('category')

    #Split the data into training and testing sets
    seed = 42
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    #Train an XGBoost classifier
    xgb = XGBClassifier(use_label_encoder=False, enable_categorical=True, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    #Set Seaborn style for better aesthetics
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    #Customize feature importance plot with seaborn styling and XGBoost plot_importance
    plot_importance(xgb, importance_type="weight", max_num_features=10, 
                    title="Top 10 Feature Importances", 
                    xlabel="F-Score", ylabel="Features", 
                    height=0.5, grid=False)
    
    #Customizing aesthetics
    plt.title('Top 10 Important Features in XGBoost Classifier', fontsize=16)
    plt.xlabel('Feature Importance (F-Score)', fontsize=12)
    plt.ylabel('Feature Names', fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    example()