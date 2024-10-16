import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import math
import shap
from sklearn.preprocessing import StandardScaler

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


def plot_categorical_features(features, X, y, title_suffix, figsize=(15, 10)):
    """Plotting the affect of categorical features on target variable

    Args:
        features (dataframe): The feaure columns
        X (dataframe): The whole dataframe
        y (series): The target variable
        figsize (tuple, optional): The size of the figure. Defaults to (15, 10).
    """
    num_features = len(features)
    
    #Dynamically determine number of rows based on number of features
    cols = 3
    rows = math.ceil(num_features / cols)
    
    plt.figure(figsize=figsize)
    
    for i, feature in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        
        #Calculate the mean of the target variable grouped by the one-hot encoded feature
        means = X.groupby(X[feature])[y.name].mean()
        
        #Create bar plot for categorical features
        sns.barplot(x=means.index, y=means.values, palette='coolwarm')
        
        #Set titles and labels
        plt.title(f'Impact of {feature} on >50K Income', fontsize=10)
        plt.xlabel(f'{feature} (0/1)', fontsize=7)
        plt.ylabel('Proportion of >50K Income', fontsize=7)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


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


def plot_logistic_regression(features, X, y, title_suffix, rows=2, cols=3, figsize=(15, 10), ):
    plt.figure(figsize=figsize)
    
    for i, feature in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        
        # Plot the logistic regression for each feature
        sns.regplot(x=X[feature], y=y, logistic=True, ci=None, 
                    scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
        
        # Set titles and labels
        plt.title(f'Logistic Regression for {feature}', fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Probability of >50K Income', fontsize=12)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def compare_to_SHAP(logit, X_train, X_nonScaled):
    """Calculates SHAP values for the provided model and training data and returns
    a DataFrame of features and their overall SHAP values.

    Args:
        logit (model): The trained model (e.g., a logistic regression or any other model).
        X_train (dataframe): The scaled or transformed training data used for model training.
        X_nonScaled (dataframe): The original, non-scaled feature data (for feature names).
        
    Returns:
        pd.DataFrame: A DataFrame containing feature names and their corresponding mean absolute SHAP values.
    """
    # Obtain SHAP values
    shapExplain = shap.Explainer(logit, X_train)
    shapValues = shapExplain(X_train)
    
    # Convert SHAP values back to original data
    shapValues.feature_names = X_nonScaled.columns.tolist()
    shapValues.data = X_nonScaled.values

    # Compute the mean absolute SHAP value for each feature
    mean_abs_shap_values = pd.DataFrame({
        'Feature': X_nonScaled.columns,
        'Mean_Abs_SHAP_Value': np.abs(shapValues.values).mean(axis=0)
    })

    # Sort by SHAP value to see the most impactful features at the top
    mean_abs_shap_values = mean_abs_shap_values.sort_values(by='Mean_Abs_SHAP_Value', ascending=False).reset_index(drop=True)
    
    return mean_abs_shap_values
    


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

    #Getting lengths of categorical features
    WClen = len(X['WorkClass'].unique())
    Elen = len(X['Education'].unique())
    MSlen = len(X['MaritalStatus'].unique())
    Rlen = len(X['Race'].unique())
    Glen = len(X['Gender'].unique())
    NClen = len(X['NativeCountry'].unique())
    Relen = len(X['Relationship'].unique())
    Olen = len(X['Occupation'].unique())

    #Encoding the nominal variables for the logistic regression
    X = pd.get_dummies(X, drop_first=True)

    #Scaling the data
    scaler = StandardScaler()
    scaler.fit(X)
    X_scale = scaler.transform(X)

    seed = 42
    np.random.seed(seed)
    X_train_scale, X_test_scale, y_train_scale, y_test_scale = train_test_split(X_scale, y, test_size=0.20, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    #Building scaled and non-scaled models
    logit_scale = LogisticRegression()
    logit_scale.fit(X_train_scale, y_train_scale)

    logit = LogisticRegression()
    logit.fit(X_train, y_train)

    #Extract log odds (coefficients) for continuous variables (only available with non-scaled data)
    log_odds = logit.coef_[0]
    features = X.columns

    shapValues = compare_to_SHAP(logit_scale, X_train_scale, X)

    #Compute the contribution of each feature for each prediction
    contributions = X_train * log_odds

    #Compute the absolute contribution of each feature for each prediction
    abs_contributions = contributions.abs()

    #Compute the average absolute contribution across all predictions for each feature
    avg_abs_contributions = abs_contributions.mean(axis=0)

    #Combine feature names and their corresponding average absolute contributions
    feature_importance_df = pd.DataFrame({'Feature': features, 'Avg_Abs_Contribution': avg_abs_contributions})

    #Sort features by their overall importance
    feature_importance_df = feature_importance_df.sort_values(by='Avg_Abs_Contribution', ascending=False)

    comparison_df = feature_importance_df.merge(shapValues, on='Feature')
    print(comparison_df)

    print("Overall importance (average absolute contribution) for each feature:")
    print(feature_importance_df)

    logistic_results = cross_validation_evaluation(logit, X_train, y_train, cv=5)
    visualize_results(logistic_results)

    #We can plot the continous features and logstic curves using seaborn regplot
    plot_logistic_regression(X_train.columns[:6], X_train, y_train, title_suffix="(1/3)")

    #Plot categorical features
    plot_categorical_features(X_train.columns[6:6 + WClen - 1], pd.concat([X_train, y_train], axis=1), y_train, title_suffix="Working Class")
    plot_categorical_features(X_train.columns[6 + WClen - 1: 6 + WClen - 1 + Elen - 1], pd.concat([X_train, y_train], axis=1), y_train, title_suffix="Education")
    plot_categorical_features(X_train.columns[29:29 + MSlen - 1], pd.concat([X_train, y_train], axis=1), y_train, title_suffix="Marital Status")
    plot_categorical_features(X_train.columns[35:35 + Olen - 1], pd.concat([X_train, y_train], axis=1), y_train, title_suffix="Occupation")
    plot_categorical_features(X_train.columns[49:49 + Relen - 1], pd.concat([X_train, y_train], axis=1), y_train, title_suffix="Relationhip")
    plot_categorical_features(X_train.columns[54:54 + Rlen - 1], pd.concat([X_train, y_train], axis=1), y_train, title_suffix="Race")
    #Only half of Native Countries because there are so many
    plot_categorical_features(X_train.columns[58:58 + (NClen // 4)], pd.concat([X_train, y_train], axis=1), y_train, title_suffix="Native Country")



if __name__ == '__main__':
    example()