import pandas as pd
import numpy as np
from math import pi
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model


class DataLoader:
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
        self.spot_price = self.data['SpotPriceDKK'].replace(0, np.nan).ffill().fillna(self.data['SpotPriceDKK'].mean())

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

class SARIMAWrapper:
    def __init__(self, order, seasonal_order, enforce_stationarity=True, enforce_invertibility=True):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self.results = None

    def fit(self, train_data, train_exog):
        self.model = SARIMAX(
            train_data,
            exog=train_exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )
        self.results = self.model.fit()
        return self.results

    def forecast_one_step(self, current_time, exog_row):
        forecast_result = self.results.get_forecast(steps=1, exog=exog_row)
        return forecast_result.predicted_mean.iloc[0]

    def bulk_forecast(self, forecast_index, exog_features_reduced, actual_data):
        forecasts = []
        last_model_time = self.results.model.data.orig_endog.index[-1]

        for current_time in forecast_index:
            # Check if we can update the model up to 24 hours before current_time
            data_end_time = current_time - pd.Timedelta(hours=24)

            if data_end_time > last_model_time:
                new_data = actual_data.loc[(last_model_time + pd.Timedelta(hours=1)):data_end_time]

                # If there's no new data to append, move on
                if not new_data.empty:
                    # Get corresponding exogenous data for the same time range
                    # Make sure exog_features_reduced covers this period
                    new_exog = exog_features_reduced.loc[new_data.index, self.results.model.data.orig_exog.columns]

                    # Append both new_data and its corresponding exog data
                    self.results = self.results.append(new_data, exog=new_exog, refit=False)
                    last_model_time = data_end_time

            # Always use exogenous data from 24 hours prior to current_time
            past_time = current_time - pd.Timedelta(hours=24)
            if past_time not in exog_features_reduced.index:
                raise ValueError(f"No exogenous data available for {past_time}.")
            current_exog = exog_features_reduced.loc[[past_time], self.results.model.data.orig_exog.columns]

            # Forecast one step ahead at current_time
            forecast_result = self.results.get_forecast(steps=1, exog=current_exog)
            forecast_value = forecast_result.predicted_mean.iloc[0]
            forecasts.append(forecast_value)

        return pd.Series(forecasts, index=forecast_index)




from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

class ARIMAWrapper:
    def __init__(self, order):
        """
        Initialize the ARIMA model wrapper.
        
        Parameters:
        - order: tuple (p, d, q) for the ARIMA model.
        """
        self.order = order
        self.model = None
        self.results = None

    def fit(self, train_data):
        """
        Fit the ARIMA model to the training data.
        
        Parameters:
        - train_data: Training time series data.
        """
        self.model = ARIMA(train_data, order=self.order)
        self.results = self.model.fit()
        return self.results

    def forecast_one_step(self):
        """
        Forecast one step ahead.
        Returns the predicted mean for the next time step.
        """
        forecast_result = self.results.get_forecast(steps=1)
        return forecast_result.predicted_mean.iloc[0]

    def bulk_forecast(self, forecast_index):
        """
        Perform bulk forecasting for a given time index.

        Parameters:
        - forecast_index: pd.DatetimeIndex for the forecast period.

        Returns:
        - pd.Series of forecasted values.
        """
        forecasts = []
        history = list(self.model.endog)  # Copy the training data into a list for updates

        for current_time in forecast_index:
            try:
                # Convert history to a clean numeric array
                clean_history = np.array(history, dtype=float)
                
                # Fit a temporary ARIMA model
                temp_model = ARIMA(clean_history, order=self.order)
                temp_results = temp_model.fit()
                
                # Forecast the next time step
                forecast_value = temp_results.forecast(steps=1)[0]
                forecasts.append(forecast_value)

                # Update history with the forecasted value
                history.append(forecast_value)

            except Exception as e:
                print(f"Error encountered at {current_time}: {e}")
                forecasts.append(np.nan)


        return pd.Series(forecasts, index=forecast_index)


    




class GARCHWrapper:
    def __init__(self, p=1, q=1, dist='normal', mean='AR', lags=1):
        self.p = p
        self.q = q
        self.dist = dist
        self.mean = mean
        self.lags = lags
        self.model = None
        self.results = None

    def fit(self, returns):
        self.model = arch_model(returns, mean=self.mean, lags=self.lags, vol='GARCH', p=self.p, q=self.q, dist=self.dist)
        self.results = self.model.fit(disp='off')
        return self.results

    def forecast_one_step(self):
        if self.results is None:
            raise ValueError("Model must be fitted before forecasting.")
        fc = self.results.forecast(horizon=1)
        mean_fc = fc.mean.iloc[-1, 0]
        return mean_fc