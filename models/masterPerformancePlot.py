from sklearn import svm
from logisticregression import cross_validation_evaluation, visualize_results
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def visualize_results(models_results):
    """
    Visualize the evaluation metrics of different models using a grouped bar chart.
    
    Args:
        models_results: A dictionary where keys are model names and values are dictionaries
                        of evaluation metrics.
    """
    #Convert models_results into a DataFrame for easier plotting

    new_dicts = []

    for model_name, metrics_dict in models_results.items():
        # Initialize a new dictionary for each model's results
        new_dict = {}
        new_dict['model'] = model_name  # Add the model name
        
        # Add all the metrics for this model
        for metric_name, value in metrics_dict.items():
            new_dict[metric_name] = value  # Add each metric as a key-value pair
        
        new_dicts.append(new_dict)  # Collect this model's results into the list

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(new_dicts)

   # Reshape the DataFrame from wide to long format
    melted_df = pd.melt(results_df, id_vars=['model'], 
                        var_name='Metric', value_name='Score')

    # Create the grouped bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Score', hue='model', data=melted_df, palette='coolwarm')

    # Customize the chart
    plt.title('Model Evaluation Metrics Comparison', fontsize=16)
    plt.xlabel('Evaluation Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(title='Model', fontsize=10, title_fontsize='13')
    plt.tight_layout()
    plt.show()


def master_results(df):
    """This method compiles all of the results for the models and visualizes them
    """

    df.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]

    X = df.iloc[:, :-1]
    y = (df.iloc[:, -1] == " >50K").astype(int)

    #Collect results from different models
    models_results = {
        'Logistic Regression': logistic_results(X, y),
        'Neural Network': train_network(),
        'SVM': svm_results(X, y),
        'XGBoost': xgb_results(X, y),
        'EBM': ebm_results(X, y)
    }

    #Call the visualization function
    visualize_results(models_results)


def ebm_results(X, y):
    """Get the results from the EBM model

    Args:
        X (dataframe): The features
        y (series): The target variable

    Returns:
        dict: A dictionary of evaluation metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    ebm = ExplainableBoostingClassifier()
    ebm.fit(X_train, y_train)

    #Perform cross-validation on EBM
    ebm_results = cross_validation_evaluation(ebm, X_train, y_train, cv=5)

    return ebm_results


def xgb_results(X, y):
    """Get the results from the XGBoost model

    Args:
        X (dataframe): The features
        y (series): The target variable

    Returns:
        dict: A dictionary of evaluation metrics
    """

    #Changing categorical type from 'Object' to 'Categorical'
    categorical_columns = ["WorkClass", "Education", "MaritalStatus", "Occupation",
                           "Relationship", "Race", "Gender", "NativeCountry"]
    for col in categorical_columns:
        X[col] = X[col].astype('category')

    #Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    #Train an XGBoost classifier
    xgb = XGBClassifier(use_label_encoder=False, enable_categorical=True, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_results = cross_validation_evaluation(xgb, X_train, y_train, cv=5)

    return xgb_results


def svm_results(X, y):
    """Get the results from the SVM model

    Args:
        X (dataframe): The features
        y (series): The target variable

    Returns:
        dict: A dictionary of evaluation metrics
    """

    #Encoding the nominal variables for the support vector machine
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    svm_classifier = svm.SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)

    svm_results = cross_validation_evaluation(svm_classifier, X_train, y_train, cv=5)

    return svm_results


def logistic_results(X, y):
    """Get the results from the EBM model

    Args:
        X (dataframe): The features
        y (series): The target variable

    Returns:
        dict: A dictionary of evaluation metrics
    """

    #Encoding the nominal variables for the logistic regression
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    logit = LogisticRegression()
    logit.fit(X_train, y_train)

    logistic_results = cross_validation_evaluation(logit, X_train, y_train, cv=5)

    return logistic_results


class FeedForward(nn.Module):
    """This is a PyTorch neural network that classifies whether the individual makes over 50k."""

    def __init__(self, input_size, layer1_size, layer2_size, output_size):
        """Layers put together in the constructor. Input of layer n is the output of layer n-1

        Args:
            input_size (int): The input shape of the features
            layer1_size (int): A hidden layer
            layer2_size (int): A hidden layer
            output_size (int): 1 for binary classification
        """
        #Call the parent constructor
        super(FeedForward, self).__init__()
        
        #Define the three layers: input -> hidden1 -> hidden2 -> output
        self.layer1 = nn.Linear(input_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.output_layer = nn.Linear(layer2_size, output_size)
        
        #Define a non-linearity (ReLU in this case)
        self.relu = nn.ReLU()

    def forward(self, x):
        #Pass data through the layers and apply non-linearities
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)  
        return x

#Train the neural network, output size 1 for binary classification
def train_network(layer1_size=256, layer2_size=128, output_size=1, num_epochs=1000):
    
    X_train, X_test, y_train, y_test, input_size = neural_network_preprocess()

    #Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    #Create an instance of the FeedForward neural network
    model = FeedForward(input_size, layer1_size, layer2_size, output_size)

    #Define the loss function (binary cross-entropy with logits)
    criterion = nn.BCEWithLogitsLoss()

    #Define the optimizer (Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Training loop
    for epoch in range(num_epochs):
        model.train()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        #Backward pass and optimization
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    #Testing the model
    results = neural_network_results(model, X_test, y_test)
    return results


def neural_network_preprocess():
    """Preprocessing adult dataset for neural net

    Returns:
        Pandas: Objects for training
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

    #Encoding nominal variables and ensuring compatibility with PyTorch
    X = pd.get_dummies(X, drop_first=True)
    #Setting all variables to float32 type with the astype() function
    X = X.astype(np.float32)

    #Scaling features with the standard scaler module from scikit-learn, this is needed for neural net optimization gradients
    sc = StandardScaler()
    X = sc.fit_transform(X)

    #Obtaining train test split
    seed = 42
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    #Converting to numpy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, X_test, y_train, y_test, X.shape[1]


def neural_network_results(model, X_test, y_test):
    """
    Evaluates the neural network on the test set and computes evaluation metrics.
    
    Args:
        model: The PyTorch model.
        X_test: The test features (torch.Tensor).
        y_test: The test labels (torch.Tensor).
    
    Returns:
        A dictionary of evaluation metrics.
    """
    #Set model to evaluation mode
    model.eval()
    
    #If not training the model (inference) set to no_grad for better efficiency
    with torch.no_grad():

        #Forward pass to get predictions
        outputs = model(X_test)
        
        #Apply sigmoid activation function to get probabilities
        probs = torch.sigmoid(outputs)
        
        #Convert probabilities to binary predictions (0 or 1) using a threshold of 0.5
        predictions = (probs >= 0.5).int()

    #Convert to numpy arrays for metric calculation
    y_test_np = y_test.cpu().numpy()
    predictions_np = predictions.cpu().numpy()

    #Calculate metrics
    metrics = {
        'f1_score': f1_score(y_test_np, predictions_np),
        'precision': precision_score(y_test_np, predictions_np),
        'recall': recall_score(y_test_np, predictions_np),
        'accuracy': accuracy_score(y_test_np, predictions_np),
        'roc_auc': roc_auc_score(y_test_np, predictions_np)
    }

    return metrics



if __name__ == '__main__':
    #Setting master seed for reproducible results
    seed = 42
    np.random.seed(seed)

    df = pd.read_csv(r"C:\Users\jakem\Downloads\adult\adult.data")

    master_results(df)


