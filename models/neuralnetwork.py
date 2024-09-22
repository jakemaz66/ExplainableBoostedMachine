import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import torch.optim as optim

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler


def example():
    """This function runs a test example of a PyTorch neural net 
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


def evaluate_nn(model, X_test, y_test):
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
        'F1-Score': f1_score(y_test_np, predictions_np),
        'Precision': precision_score(y_test_np, predictions_np),
        'Recall': recall_score(y_test_np, predictions_np),
        'Accuracy': accuracy_score(y_test_np, predictions_np),
    }

    return metrics


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
    
    X_train, X_test, y_train, y_test, input_size = example()

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
    results = evaluate_nn(model, X_test, y_test)
    visualize_results(results)


if __name__ == '__main__':
    train_network()


    """Notes
    1. Right now rthe model is only prediction the same probability for each row. The loss isn't going down
    2. You definitely have to normalize data with neural networks

    """