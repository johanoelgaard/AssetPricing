import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# create data class
class ElectricityDataset(Dataset):
    def __init__(self, features, targets, seq_length=24):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.targets) - self.seq_length

    def __getitem__(self, idx):
        X = self.features[idx:idx+self.seq_length]
        y = self.targets[idx+self.seq_length]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# define neural network
class PricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(PricePredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # initialize hidden and cell states
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        # LSTM output
        out, _ = self.lstm(x, (h0, c0))

        # get output of the last time step
        out = self.fc(out[:, -1, :])
        return out


# define function for L1 regularization
def l1_regularization(model, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm