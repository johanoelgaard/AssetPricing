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
class dataset(Dataset):
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
class LSTMmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMmodel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # fully connected output layer
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


# moving to GPU if available (Metal)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# set random seed for reproducibility
torch.manual_seed(2024)
np.random.seed(2024)

# path to the CSV file
path = '../../data/fulldata.csv'

data = pd.read_csv(path)

print(data.head())
print(data.shape)

data['from'] = pd.to_datetime(data['from'])

# drop to column
data = data.drop(columns=['to'])

# get old prices 
lag_hours = [0, 
            -1, -2, -3, -4, -5, -6, -24, -48, -72, -96, -120, -144, -168
             ]
for lag in lag_hours:
    data[f'price_lag_{lag}'] = data['SpotPriceDKK'].shift(lag)

# offset price by 1 day
data['SpotPriceDKK'] = data['SpotPriceDKK'].shift(24)
# drop the first 24 rows
data = data.dropna()

# offset the from date by 1 day to match the price
data['from'] = data['from'] + pd.DateOffset(days=1)

# extract time features
data['hour'] = data['from'].dt.hour
data['day_of_week'] = data['from'].dt.dayofweek
data['month'] = data['from'].dt.month

# cyclical encoding for hour, day, and month
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# resort data to be ascending
data = data.sort_values('from')

print(data.head())
print(data.shape)

# extract column names
cols = data.columns.tolist()

# select features and target variable
all_features = cols[2:]
target = cols[:1]

print(f'Count of features before interaction terms: {len(all_features)}')
print(f'Target variable: {target}')

# training data: until July 2023
train_data = data[data['from'] < '2023-08-01']

# validation data: July 2023 to December 2023
val_data = data[(data['from'] >= '2023-08-01') & (data['from'] < '2024-08-01')]

# Test data: 2024 and beyond
test_data = data[data['from'] >= '2024-08-01']

X_train = train_data[all_features].values
X_val = val_data[all_features].values
X_test = test_data[all_features].values

# initialize the scaler
scaler = StandardScaler()

# fit the scaler on the training features and transform
train_features_scaled = scaler.fit_transform(X_train)

# transform the validation and test features using the same scaler
val_features_scaled = scaler.transform(X_val)
test_features_scaled = scaler.transform(X_test)


# extract target values
train_targets = train_data[target].values
val_targets = val_data[target].values
test_targets = test_data[target].values


# hyperparameters
seq_length = 24  # Use past 24 hours to form a sequence
batch_size = 256
input_dim = train_features_scaled.shape[1]
hidden_dim = 128
layer_dim = 2
output_dim = 1
learning_rate = 0.00075

# L1 regularization parameter
lambda_l1 = 1.5 #1e-5  # Adjust based on desired regularization strength


# create datasets
train_dataset = dataset(train_features_scaled, train_targets, seq_length)
val_dataset = dataset(val_features_scaled, val_targets, seq_length)
test_dataset = dataset(test_features_scaled, test_targets, seq_length)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# initialize the model
model = LSTMmodel(input_dim, hidden_dim, layer_dim, output_dim)

# loss evaluation function
criterion = nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# training the model
num_epochs = 100
patience = 10  # for early stopping
best_loss = np.inf
counter = 0

model.to(device)


for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        # forward pass
        outputs = model(X_batch)

        mse_loss = criterion(outputs.squeeze(), y_batch.squeeze())

        # L1 regularization
        l1_loss = l1_regularization(model, lambda_l1)

        # calc total loss
        loss = mse_loss + l1_loss

        # backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # average training loss
    avg_epoch_loss = epoch_loss / len(train_loader)

    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            mse_loss = criterion(outputs.squeeze(), y_batch.squeeze())
            l1_loss = l1_regularization(model, lambda_l1)
            loss = mse_loss + l1_loss
            val_losses.append(loss.item())

    avg_val_loss = np.mean(val_losses)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    # early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        counter = 0
        # save the best model
        torch.save(model.state_dict(), 'output/best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# load the best model
model.load_state_dict(torch.load('output/best_model.pth'))

model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        predictions.extend(outputs.squeeze().tolist())
        actuals.extend(y_batch.tolist())

rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f'Performance on test data (2024):\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR^2 Score: {r2:.4f}')


# convert 'predictions' and 'actuals' to numpy arrays
predictions = np.array(predictions)
actuals = np.array(actuals)

# extract 'from' timestamps from test_data, adjusted for seq_length
# since the dataset uses sequences, the first 'seq_length' targets are not included in 'actuals' and 'predictions'
test_timestamps = test_data['from'].values[seq_length:]

# ensure lengths match
min_length = min(len(test_timestamps), len(actuals), len(predictions))
test_timestamps = test_timestamps[:min_length]
actuals = actuals[:min_length]
predictions = predictions[:min_length]

# create the plot
plt.figure(figsize=(15, 7))
plt.plot(test_timestamps, actuals, label='Actual Prices', color='blue', alpha=0.7)
plt.plot(test_timestamps, predictions, label='Predicted Prices', color='red', alpha=0.7)
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Electricity Price')
plt.title('Predicted vs. Actual Electricity Prices on Test Data')

# format x-axis with date labels
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('output/lstm_predicted_vs_actual_prices2024.png')

plt.show()

# plot the first 500 data points
datapoints = 24*21
plt.figure(figsize=(15, 7))
plt.plot(test_timestamps[:datapoints], actuals[:datapoints], label='Actual Prices', color='blue', alpha=0.7)
plt.plot(test_timestamps[:datapoints], predictions[:datapoints], label='Predicted Prices', color='red', alpha=0.7)
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Electricity Price')
plt.title(f'Predicted vs. Actual Electricity Prices (Last {datapoints//24} Days)')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(f'output/lstm_predicted_vs_actual_prices_last{datapoints//24}days.png')
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(actuals, predictions, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs. Actual Electricity Prices')

# Plot a diagonal line for reference
min_price = min(actuals.min(), predictions.min())
max_price = max(actuals.max(), predictions.max())
plt.plot([min_price, max_price], [min_price, max_price], 'k--', lw=2)
plt.tight_layout()

plt.savefig('output/lstm_predicted_vs_actual_prices_scatter.png')
plt.show()
