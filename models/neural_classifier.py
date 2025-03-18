import torch
from torch.nn.functional import relu, sigmoid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import numpy as np


class NeuralBinaryClassifier(torch.nn.Module):
    def __init__(self, num_epochs = 8, neurons_in_layer=16):
        super().__init__()
        self.num_epochs = 8
        self.neurons_in_layer = neurons_in_layer

    def forward(self, x):
        x = relu(self.fc1(x))
        x = sigmoid(self.fc2(x))
        return x

    def fit(self, X, y, sample_weight=None):
        self.fc1 = torch.nn.Linear(X.shape[1], self.neurons_in_layer)
        self.fc2 = torch.nn.Linear(self.neurons_in_layer, 1)
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(y.shape[0]) / y.shape[0]

        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        optimizer =  torch.optim.Adam(self.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss(weight=torch.tensor(sample_weight, dtype=torch.float32))
        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            outputs = self.__call__(X).squeeze()
            losses = criterion(outputs, y)
            losses.backward()
            optimizer.step()
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        pred = self.__call__(X).squeeze()
        return (pred.detach().numpy() >= 0.5)

    def get_params(self, deep=True):
        return { "num_epochs" : self.num_epochs,
                "neurons_in_layer" : self.neurons_in_layer}
    
    def set_params(self, **kwargs):
        self.num_epochs = kwargs.get('num_epochs', self.num_epochs)
        self.neurons_in_layer = kwargs.get('neurons_in_layer', self.neurons_in_layer)
        return self

    def score(self, X, y):
        return (self.predict(X)==y).sum()/X.shape[0]