import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import t
import numpy as np


# create data class
class LSTMdataset(Dataset):
    """ 
    PyTorch Dataset class for LSTM model.
    """
    def __init__(self, features, targets, seq_length):
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
    """
    LSTM model for time series forecasting.
    """
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
    """
    L1 regularization for PyTorch model.
    
    Args:
    - model: PyTorch model.
    - lambda_l1: L1 regularization parameter.
    
    Returns:
    - L1 regularization loss.
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

def l2_regularization(model, lambda_l2):
    """
    L2 regularization for PyTorch model.
    
    Args:
    - model: PyTorch model.
    - lambda_l2: L2 regularization parameter.
    
    Returns:
    - L2 regularization loss.
    """
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return lambda_l2 * l2_norm

class SARIMADataLoader:
    """
    Data loader for SARIMA model.
    """
    def __init__(self, file_path, use_temp_pca=True, use_wind_pca=True, use_fourier=True, use_oil=True, use_gas=True):
        self.file_path = file_path
        self.use_temp_pca = use_temp_pca
        self.use_wind_pca = use_wind_pca
        self.use_fourier = use_fourier
        self.use_oil = use_oil
        self.use_gas = use_gas

        self.data = None
        self.spot_price = None
        self.exog_data = None

    def load_data(self):
        # Load and clean the data
        data = pd.read_csv(self.file_path)
        data['from'] = pd.to_datetime(data['from'], utc=True).dt.tz_localize(None)
        data.set_index('from', inplace=True)
        self.data = data
        return self.data

    def preprocess_data(self):
        # Spot price: No differencing, just forward-fill zeros and NaNs
        self.spot_price = self.data['SpotPrice']

        # Build exogenous features dynamically
        exog_components = {}

        # Temperature PCA components
        if self.use_temp_pca:
            temp_pca_cols = [col for col in self.data.columns if 'temp_pca' in col]
            for col in temp_pca_cols:
                exog_components[col] = self.data[col]

        # Wind Speed PCA components
        if self.use_wind_pca:
            wind_pca_cols = [col for col in self.data.columns if 'wind_speed_pca' in col]
            for col in wind_pca_cols:
                exog_components[col] = self.data[col]

        # Fourier Features
        if self.use_fourier:
            fourier_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
            for col in fourier_cols:
                if col in self.data.columns:
                    exog_components[col] = self.data[col]

        # Oil Price
        if self.use_oil and 'oil_price' in self.data.columns:
            exog_components['oil_price'] = self.data['oil_price']

        # Gas Price
        if self.use_gas and 'gas_price' in self.data.columns:
            exog_components['gas_price'] = self.data['gas_price']

        # Combine exogenous components
        self.exog_data = pd.DataFrame(exog_components, index=self.data.index)

        # Ensure data alignment
        self.spot_price = self.spot_price.sort_index()
        self.exog_data = self.exog_data.sort_index()

        return self.spot_price, self.exog_data
    

def latex_table(models, metrics):
    """
    Generates a LaTeX table in wide format comparing models based on given metrics.

    Args:
    models (list of str): Names of the models (e.g., ['Naive', 'SARIMA', 'SARIMAX', 'LSTM']).
    metrics (dict): Dictionary with metric names as keys (e.g., 'RMSE', 'MAE') and 
                    lists of metric values as values. For p-values, use tuples where the second
                    value is the p-value (e.g., {'RMSE': [123, (101, 0.05), (95, 0.03), (85, 0.01)], ...}).

    Returns:
    str: A LaTeX table string in wide format.
    """
    # Start the table
    table = "\\begin{tabular}{l" + "c" * len(models) + "}\n\hline\hline \\\\ [-1.8ex]\n"

    # Add the header
    table += " & " + " & ".join(models) + " \\\\ \n \hline \n"

    # Add rows for each metric
    for metric, values in metrics.items():
        # Main metric values row
        row = f"{metric} & "
        row_values = []
        for value in values:
            if isinstance(value, tuple):  # If it's a tuple, extract the main value
                main_value, _ = value
                row_values.append(f"{main_value:.2f}")
            else:
                row_values.append(f"{value:.2f}")
        row += " & ".join(row_values) + " \\\\ \n"
        table += row

        # P-values row
        p_row = "" if metric.strip() == "" else " & "
        p_values = []
        for value in values:
            if isinstance(value, tuple):  # If it's a tuple, extract the p-value
                _, p_value = value
                p_values.append(f"({p_value:.3f})")
            else:
                p_values.append("-")
        p_row += " & ".join(p_values) + " \\\\ \n"
        table += p_row

    # Close the table
    table += "\hline\hline\n\end{tabular}"

    return table

def plot_forecasts(
    timestamps, 
    actuals, 
    naive, 
    sarima_forecast, 
    sarimax_forecast, 
    lstm_forecast, 
    start_datetime, 
    end_datetime, 
    output_path=None,
    display_plot=False
):
    """
    Plots forecast comparisons against actual values in four subplots.

    Args:
    - timestamps: pandas.Series or numpy.array of datetime values.
    - actuals: pandas.Series or numpy.array of actual values.
    - naive: pandas.Series or numpy.array of naïve forecast values.
    - sarima_forecast: pandas.Series or numpy.array of SARIMA forecast values.
    - sarimax_forecast: pandas.Series or numpy.array of SARIMAX forecast values.
    - lstm_forecast: pandas.Series or numpy.array of LSTM forecast values.
    - start_datetime: str, inclusive start datetime (e.g., "2024-08-01 00:00").
    - end_datetime: str, exclusive end datetime (e.g., "2024-08-10 00:00").
    - output_path: str, path to save the output plot (default: None).
    - display_plot: bool, whether to display the plot (default: False).
    """
    # Ensure timestamps are datetime
    timestamps = pd.to_datetime(timestamps)

    # Filter the data based on the datetime range
    mask = (timestamps >= start_datetime) & (timestamps < end_datetime)
    filtered_timestamps = timestamps[mask]
    filtered_actuals = actuals[mask]
    filtered_naive = naive[mask]
    filtered_sarima_forecast = sarima_forecast[mask]
    filtered_sarimax_forecast = sarimax_forecast[mask]
    filtered_lstm_forecast = lstm_forecast[mask]

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))

    # Define models and their labels/colors
    models = [
    ("Naïve Forecast", filtered_naive, '#ff7f0e'),
    ("SARIMA Forecast", filtered_sarima_forecast, '#e377c2'),
    ("LSTM Forecast", filtered_lstm_forecast, '#d62728'),
    ("SARIMAX Forecast", filtered_sarimax_forecast, '#bcbd22'),
    ]

    # Plot each model on its own subplot
    for ax, (label, model, color) in zip(axs.flat, models):
        ax.plot(filtered_timestamps, filtered_actuals, label='Actual Prices', color='#1f77b4', alpha=0.8, linewidth=2.5)
        ax.plot(filtered_timestamps, model, label=label, color=color, alpha=0.7, linewidth=2.5)
        ax.set_ylabel('Spot Price (DKK per MWh)', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(fontsize=20)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.tick_params(axis='x', rotation=45)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save and show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Forecast plot saved to {output_path}")
    if display_plot:
        plt.show()

    plt.close()

import matplotlib.pyplot as plt

def plot_forecast_scatter(
    actuals,
    naive,
    sarima_forecast,
    sarimax_forecast,
    lstm_forecast,
    output_path=None,
    display_plot=False
):
    """
    Plots scatter plots of forecast models against actual values in a 2x2 grid.

    Args:
    - actuals: pandas.Series or numpy.array of actual values.
    - naive: pandas.Series or numpy.array of naive forecast values.
    - sarima_forecast: pandas.Series or numpy.array of SARIMA forecast values.
    - sarimax_forecast: pandas.Series or numpy.array of SARIMAX forecast values.
    - lstm_forecast: pandas.Series or numpy.array of LSTM forecast values.
    - output_path: str, path to save the output plot (default: None).
    - display_plot: bool, whether to display the plot (default: False).
    """
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # Define models and their labels
    models = [
        ("Naïve Forecast", naive),
        ("SARIMA Forecast", sarima_forecast),
        ("LSTM Forecast", lstm_forecast),
        ("SARIMAX Forecast", sarimax_forecast),
    ]

    # Determine the range for the diagonal reference line
    min_price = min(actuals.min(), naive.min(), sarima_forecast.min(), sarimax_forecast.min(), lstm_forecast.min())
    max_price = max(actuals.max(), naive.max(), sarima_forecast.max(), sarimax_forecast.max(), lstm_forecast.max())

    # Plot each model on its own subplot
    for ax, (label, forecast) in zip(axs.flat, models):
        ax.scatter(actuals, forecast, alpha=0.5)
        ax.plot([min_price, max_price], [min_price, max_price], 'k--', lw=2)  # Diagonal reference line
        ax.set_xlabel('Actual Prices', fontsize=24)
        ax.set_ylabel(label, fontsize=24)
        ax.set_title(f'{label} vs Actual Prices', fontsize=26)
        ax.tick_params(axis='both', which='major', labelsize=20)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save and show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Scatter plot saved to {output_path}")
    if display_plot:
        plt.show()

    plt.close()

def diebold_mariano_test(actuals, forecast1, forecast2, loss_function='mse', h=1):
    """
    Performs the Diebold-Mariano test for predictive accuracy.

    Args:
        actuals (array): Actual observed values.
        forecast1 (array): First forecast to compare.
        forecast2 (array): Second forecast to compare.
        loss_function (str): The loss function to use ('mse' or 'mae').
        h (int): Forecast horizon, default is 1 (single-step forecast).

    Returns:
        DM statistic and p-value.
    """
    # Compute forecast errors
    error1 = actuals - forecast1
    error2 = actuals - forecast2

    # Choose the loss function
    if loss_function == 'mse':
        diff = error1**2 - error2**2
    elif loss_function == 'mae':
        diff = np.abs(error1) - np.abs(error2)
    else:
        raise ValueError("Unsupported loss function. Use 'mse' or 'mae'.")

    # Compute mean and variance of the loss differential
    mean_diff = np.mean(diff)
    n = len(diff)
    variance_diff = np.var(diff, ddof=1)

    # Correct variance for autocorrelation if h > 1
    if h > 1:
        autocov = np.correlate(diff, diff, mode='full') / n
        variance_diff += 2 * sum(autocov[n-1:n-1+h])

    # Compute DM statistic
    dm_stat = mean_diff / np.sqrt(variance_diff / n)

    # Compute p-value
    dof = n - 1  # Degrees of freedom
    p_value = 2 * (1 - t.cdf(abs(dm_stat), df=dof))  # Two-tailed test

    return dm_stat, p_value
