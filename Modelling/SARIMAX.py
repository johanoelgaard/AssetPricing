import pandas as pd
import numpy as np
from math import pi
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model


class DataLoader:
    def __init__(self, file_path, use_temp=True, use_wind=True, use_oil=True, use_gas=True):
        self.file_path = file_path
        self.use_temp = use_temp
        self.use_wind = use_wind
        self.use_oil = use_oil
        self.use_gas = use_gas

        self.data = None
        self.spot_price = None
        self.exog_data = None

    def load_data(self):
        data = pd.read_csv(self.file_path)
        data['from'] = pd.to_datetime(data['from'], utc=True).dt.tz_localize(None)
        data.set_index('from', inplace=True)
        self.data = data
        return self.data

    def preprocess_data(self):
        self.spot_price = self.data['SpotPriceDKK'].replace(0, np.nan).ffill().fillna(self.data['SpotPriceDKK'].mean())

        exog_components = {}

        # Temperature
        if self.use_temp:
            temp_cols = [col for col in self.data.columns if 'temp' in col]
            if temp_cols:
                temperature_data = self.data[temp_cols].ffill().fillna(self.data[temp_cols].mean())
                avg_temp = temperature_data.mean(axis=1)
                exog_components['avg_temp'] = avg_temp

        # Wind (if use_wind=True)
        if self.use_wind:
            # Same logic as before for wind direction and speed
            direction_map = {
               'N': 0,
               'NE': np.pi/4,
               'E': np.pi/2,
               'SE': 3*np.pi/4,
               'S': np.pi,
               'SW': 5*np.pi/4,
               'W': 3*np.pi/2,
               'NW': 7*np.pi/4
            }

            wind_speed_cols = [col for col in self.data.columns if 'wind_speed_' in col]
            wind_dir_cols = [col for col in self.data.columns if 'wind_dir_' in col]

            if wind_speed_cols and wind_dir_cols:
                # Fill missing
                self.data[wind_speed_cols] = self.data[wind_speed_cols].ffill().fillna(self.data[wind_speed_cols].mean())
                self.data[wind_dir_cols] = self.data[wind_dir_cols].ffill().fillna(False)
                
                municipalities = [col.split('wind_speed_')[-1] for col in wind_speed_cols]
                wind_x_all = []
                wind_y_all = []
                
                for muni in municipalities:
                    muni_dir_cols = [c for c in wind_dir_cols if c.endswith('_' + muni)]
                    muni_speed_col = 'wind_speed_' + muni

                    directions_df = self.data[muni_dir_cols].astype(int)
                    dir_matrix = directions_df.values
                    true_count = dir_matrix.sum(axis=1)
                    # Assuming one True per row
                    direction_indices = dir_matrix.argmax(axis=1)
                    chosen_dirs = [muni_dir_cols[i].split('wind_dir_')[1].split('_')[0] for i in direction_indices]
                    angles = [direction_map[d] for d in chosen_dirs]
                    speeds = self.data[muni_speed_col].values
                    muni_wind_x = speeds * np.cos(angles)
                    muni_wind_y = speeds * np.sin(angles)
                    wind_x_all.append(muni_wind_x)
                    wind_y_all.append(muni_wind_y)

                if municipalities:
                    wind_x_all = np.array(wind_x_all)
                    wind_y_all = np.array(wind_y_all)
                    avg_wind_x = wind_x_all.mean(axis=0)
                    avg_wind_y = wind_y_all.mean(axis=0)
                    exog_components['avg_wind_x'] = avg_wind_x
                    exog_components['avg_wind_y'] = avg_wind_y

        # Oil price
        if self.use_oil and 'oil_price' in self.data.columns:
            oil_price_data = self.data['oil_price'].ffill().fillna(self.data['oil_price'].mean())
            exog_components['oil_price'] = oil_price_data

        # Gas price
        if self.use_gas and 'gas_price' in self.data.columns:
            gas_price_data = self.data['gas_price'].ffill().fillna(self.data['gas_price'].mean())
            exog_components['gas_price'] = gas_price_data

        self.exog_data = pd.DataFrame(exog_components, index=self.data.index)
        self.spot_price = self.spot_price.sort_index()
        self.exog_data = self.exog_data.sort_index()
        return self.spot_price, self.exog_data


class FeatureEngineer:
    def __init__(self, use_fourier=False, use_pca=False, pca_components=0.95):
        self.use_fourier = use_fourier
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.scaler = None
        self.pca = None

    def add_fourier_terms(self, spot_price, exog_data):
        if not self.use_fourier:
            return exog_data
        
        hours_in_day = 24
        hours_in_week = 24 * 7
        hour_of_day = spot_price.index.hour
        hour_of_week = spot_price.index.dayofweek * 24 + spot_price.index.hour

        x_fourier = pd.DataFrame(index=spot_price.index)
        x_fourier['sin_hourly'] = np.sin(2 * pi * hour_of_day / hours_in_day)
        x_fourier['cos_hourly'] = np.cos(2 * pi * hour_of_day / hours_in_day)
        x_fourier['sin_weekly'] = np.sin(2 * pi * hour_of_week / hours_in_week)
        x_fourier['cos_weekly'] = np.cos(2 * pi * hour_of_week / hours_in_week)
        x_fourier['time_trend'] = np.arange(len(x_fourier))

        return pd.concat([exog_data, x_fourier], axis=1)

    def apply_pca(self, exog_data):
        if not self.use_pca:
            return exog_data
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(exog_data)
        self.pca = PCA(n_components=self.pca_components)
        pca_features = self.pca.fit_transform(scaled)
        return pd.DataFrame(pca_features, index=exog_data.index)

    def transform(self, spot_price, exog_data):
        # Add Fourier terms if requested
        exog_data = self.add_fourier_terms(spot_price, exog_data)
        # Apply PCA if requested
        exog_data = self.apply_pca(exog_data)
        return exog_data


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

    def bulk_forecast(self, forecast_index, exog_features_reduced, use_past_24h=False):
        forecasts = []
        for current_time in forecast_index:
            if use_past_24h:
                past_time = current_time - pd.Timedelta(hours=24)
                if past_time not in exog_features_reduced.index:
                    raise ValueError(f"No exogenous data for {past_time}.")
                current_exog = exog_features_reduced.loc[[past_time]]
            else:
                # If not using past 24h logic, assume direct exog is available
                current_exog = exog_features_reduced.loc[[current_time]]

            forecast_value = self.forecast_one_step(current_time, current_exog)
            forecasts.append(forecast_value)
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