import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import DMatrix, plot_importance

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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
    
    #Customize feature importance plot with seaborn styling and XGBoost plot_importance
    plot_importance(xgb, importance_type="weight", max_num_features=15, 
                    title="Top 15 Feature Importances", 
                    xlabel="F-Score", ylabel="Features", 
                    height=0.5, grid=False)
    
    #Customizing aesthetics
    plt.title('Top 15 Important Features in XGBoost Classifier', fontsize=16)
    plt.xlabel('Feature Importance (F-Score)', fontsize=12)
    plt.ylabel('Feature Names', fontsize=12)
    plt.tight_layout()
    plt.show()

    #Cross validating and visualizing results
    xgb_results = cross_validation_evaluation(xgb, X_train, y_train, cv=5)
    visualize_results(xgb_results)

if __name__ == '__main__':
    example()