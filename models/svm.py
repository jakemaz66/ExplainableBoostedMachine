import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns

import math

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def cross_validation_evaluation(model, X, y, cv=5, random_state=42):
    """
    Perform cross-validation and evaluate the model on multiple metrics.
    
    Args:
        model: The machine learning model
        X: The input features.
        y: The target variable.
        cv: The number of cross-validation folds.
        random_state: Seed for reproducibility.
    
    Returns:
        Dictionary of evaluation metrics from cross-validation.
    """
    
    #Define the evaluation metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score)
    }

    #Perform cross-validation with StratifiedKFold for balanced classes
    cv_results = cross_validate(
        model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring=scoring, return_train_score=False
    )
    
    #Display the average of each metric across all folds
    results = {
        'accuracy': np.mean(cv_results['test_accuracy']),
        'precision': np.mean(cv_results['test_precision']),
        'recall': np.mean(cv_results['test_recall']),
        'f1_score': np.mean(cv_results['test_f1']),
        'roc_auc': np.mean(cv_results['test_roc_auc'])
    }
    
    return results

def visualize_results(models_results):
    """
    Visualize the evaluation metrics of different models using a grouped bar chart.
    
    Args:
        models_results: A dictionary where keys are model names and values are dictionaries
                        of evaluation metrics.
    """
    #Convert the results into a DataFrame for easier plotting
    models_results['Index'] = np.arange(len(models_results))
    results_df = pd.DataFrame(models_results, index=models_results['Index']).T.reset_index()
    
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='index', y=0, data=results_df, palette='coolwarm')
    
    plt.title('Model Evaluation Metrics Comparison', fontsize=16)
    plt.xlabel('Evaluation Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(title='Model', fontsize=10, title_fontsize='13')
    plt.tight_layout()
    plt.show()


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

    #Encoding the nominal variables for the support vector machine
    X = pd.get_dummies(X, drop_first=True)
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

    #Cross validating and visualizing results
    svm_results = cross_validation_evaluation(svm_classifier, X_train, y_train, cv=5)
    visualize_results(svm_results)


if __name__ == '__main__':
    example()

    """Notes

    1. Support vector machines take a REALLY long time to train, they have to do kernel functions
    """