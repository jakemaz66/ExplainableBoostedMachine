import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import torch.optim as optim


class FeedForward(nn.Module):
    """This is a PyTorch neural network that takes in the features of the dataset
       and classifies whether the individual makes over 50k.
    """

    def __init__(self, input_size, layer1_size, layer2_size, output_size):
        #Calling the parent constructor
        super(FeedForward, self).__init__()
        
        #Define the three layers: input -> hidden1 -> hidden2 -> output
        self.layer1 = nn.Linear(input_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.output_layer = nn.Linear(layer2_size, output_size)
        
        #Define a non-linearity (ReLU in this case)
        self.relu = nn.ReLU()
        
        #Use sigmoid for binary classification at the output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #Pass data through the layers and apply non-linearities
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x


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

    #Convert to torch tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)

    #Binary classification shape
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  

    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    return X_train, X_test, y_train, y_test, X.shape[1]

#Train the neural network, output size 1 for binary classification
def train_network(layer1_size=64, layer2_size=32, output_size=1, num_epochs = 100):
    X_train, X_test, y_train, y_test, input_size = example()

    #Create an instance of the FeedForward neural network
    model = FeedForward(input_size, layer1_size, layer2_size, output_size)

    #Define the loss function (binary cross entropy)
    criterion = nn.BCELoss()

    #Define the optimizer (Stochastic Gradient Descent)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #Training loop
    for epoch in range(num_epochs):
        model.train()
        
        outputs = model(X_train)
        
        loss = criterion(outputs, y_train)
        
        #Backward pass and optimization
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()       

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    #Testing the model
    model.eval()
    with torch.no_grad():
        predicted = model(X_test)
        predicted = (predicted > 0.5).float()  #Convert probabilities to binary output
        accuracy = (predicted.eq(y_test).sum() / y_test.shape[0]).item()
        print(f'Accuracy on test data: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    train_network()