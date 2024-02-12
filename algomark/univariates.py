import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class TimeSeriesForecaster:
    def __init__(self, ihs_data, vol_data):
        """
        Initialize the class with time series data.
        :param data: pandas DataFrame with time series data
        """

        self.ihs_data = ihs_data
        self.vol_data = vol_data

    def data_checks(self):
        """ The module performs data checks on the input data."""
        pass

    def data_imputation(self):
        pass

    def preprocess_data(self, model_name, last_forecasting_year):
        """
        Preprocess data based on the model requirement.
        :param model_name: Name of the model to preprocess data for
        :return: Preprocessed data
        """
        # Implement preprocessing logic for each model
        # For simplicity, assuming the data is already in the required format for now

        self.vol_data['Date'] = pd.to_datetime(self.vol_data['Date'], "%Y-%m-%d")
        self.vol_data.rename({'Date': 'Period'}, axis=1, inplace=True)

        self.ihs_data['Variable'] = self.ihs_data['Concept'] + '_' + self.ihs_data['Frequency']
        self.ihs_data['Period'] = pd.to_datetime(self.ihs_data['Period'])
        # try:
        #     self.ihs_data['Period'] = pd.to_datetime(self.ihs_data['Period'], format="%Y-%m-%d")
        # except:
        #     self.ihs_data['Period'] = pd.to_datetime(self.ihs_data['Period'], format="%m-%d-%Y")
        self.ihs_data['Year'] = self.ihs_data['Period'].dt.year  # pd.DatetimeIndex(self.ihs_data.Period).year
        self.ihs_data['Month'] = self.ihs_data['Period'].dt.month  # pd.DatetimeIndex(self.ihs_data.Period).month
        self.ihs_data['Quarter'] = self.ihs_data['Period'].dt.quarter  # pd.DatetimeIndex(self.ihs_data.Period).quarter

        self.ihs_data = self.ihs_data.drop(['Unit', 'Data Edge', 'Last Update', 'Start Date'], axis=1, errors='ignore')

        self.ihs_data = self.ihs_data[self.ihs_data['Year'] <= last_forecasting_year]
        date_list = []
        for i in self.ihs_data['Year'].unique():
            for j in self.ihs_data['Month'].unique():
                if len(str(j)) < 2:
                    date_ = str(i) + '-0' + str(j) + '-' + '01'
                else:
                    date_ = str(i) + '-' + str(j) + '-' + '01'
                date_list.append(date_)

        date_df = pd.DataFrame(date_list, columns=['Period'])
        date_df['Period'] = pd.to_datetime(date_df.Period, format="%Y-%m-%d")
        geo = pd.DataFrame(self.ihs_data['Geography'].unique(), columns=['Geography'])
        geo['key'] = 1
        date_df['key'] = 1
        date_df = date_df.merge(geo, on='key', how='outer')
        data_annual = self.ihs_data[self.ihs_data['Frequency'] == 'Annual'].reset_index(drop=True)
        data_quarterly = self.ihs_data[self.ihs_data['Frequency'] == 'Quarterly'].reset_index(drop=True)
        data_monthly = self.ihs_data[self.ihs_data['Frequency'] == 'Monthly'].reset_index(drop=True)

        data_pivot_annual = data_annual.pivot_table(index=['Geography', 'Period', 'Year', 'Month', 'Quarter'],
                                                        columns=['Variable'], values='Value', aggfunc="mean")

        data_pivot_quarterly = data_quarterly.pivot_table(
            index=['Geography', 'Period', 'Year', 'Month', 'Quarter'], columns=['Variable'], values='Value',
            aggfunc="mean")

        data_pivot_monthly = data_monthly.pivot_table(index=['Geography', 'Period', 'Year', 'Month', 'Quarter'],
                                                              columns=['Variable'], values='Value', aggfunc="mean")
        data_pivot_monthly = date_df.merge(data_pivot_monthly, on=['Period', 'Geography'], how='left')
        data_pivot_annual = data_pivot_annual.reset_index(drop=False)
        data_pivot_quarterly = data_pivot_quarterly.reset_index(drop=False)
        data_pivot_monthly = data_pivot_monthly.reset_index(drop=False)
        data_pivot_monthly['Year'] = pd.DatetimeIndex(data_pivot_monthly.Period).year
        data_pivot_monthly['Month'] = pd.DatetimeIndex(data_pivot_monthly.Period).month
        data_pivot_monthly['Quarter'] = pd.DatetimeIndex(data_pivot_monthly.Period).quarter

        data_pivot_monthly = data_pivot_monthly.merge(
            data_pivot_annual.drop(['Month', 'Quarter', 'Period'], axis=1, errors='ignore'), on=['Geography', 'Year'], how='outer')
        data_pivot_monthly = data_pivot_monthly.merge(
            data_pivot_quarterly.drop(['Month', 'Period'], axis=1, errors='ignore'), on=['Geography', 'Year', 'Quarter'], how='outer')
        # del(data,data_annual,data_quarterly,data_quarterly,data_pivot_annual,data_pivot_quarterly)
        data_pivot_monthly = data_pivot_monthly.drop(['key', 'Year', 'Month', 'Quarter', 'index'], axis=1).reset_index(drop=True)

        self.data = self.vol_data.merge(data_pivot_monthly, on = "Period", how='left')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[[i for i in self.data.columns if i!= "Volume"]], self.data["Volume"], test_size=0.2, random_state=42)

        return self.data



    def arima_forecast(self, order):
        """
        Perform time series forecasting using the ARIMA model.
        :param order: The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters
        :return: Forecasted values
        """
        model = ARIMA(self.y_train, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=12)  # Forecasting next 12 periods
        return forecast

    def exponential_smoothing_forecast(self, seasonal_periods):
        """
        Perform time series forecasting using Exponential Smoothing.
        :param seasonal_periods: The number of time steps in a seasonal period
        :return: Forecasted values
        """
        model = ExponentialSmoothing(self.data, seasonal='add', seasonal_periods=seasonal_periods)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=5)
        return forecast

    def linear_regression_forecast(self):
        """
        Perform time series forecasting using Linear Regression.
        :return: Forecasted values
        """
        # For Linear Regression, transform time series data to a supervised learning problem
        # This requires more complex data preprocessing (e.g., creating lag features)
        # Assuming data is preprocessed

        # Placeholder for actual implementation

        model = LinearRegression()
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
        return forecast

    def prophet_forecast(self):
        """
        Perform time series forecasting using Prophet.
        :return: Forecasted values
        """
        model = Prophet()
        model.fit(self.data)
        future = model.make_future_dataframe(periods=5, freq='M')
        forecast = model.predict(future)
        return forecast

    def xgboost_forecast(self):
        """
        Perform time series forecasting using XGBoost.
        :return: Forecasted values
        """
        # For XGBoost, similar to Linear Regression, the time series data needs to be transformed
        # Placeholder for actual implementation
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.data, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
        return forecast

# Note: Function calls are commented out to prevent execution in this environment.
# Usage Example:

ihs_data = pd.read_csv('sample_data/sample_macro.csv')
vol_data = pd.read_excel('sample_data/sample_input.xlsx')
forecaster = TimeSeriesForecaster(ihs_data, vol_data)
data = forecaster.preprocess_data("linear", 2025)
arima_forecast = forecaster.arima_forecast(order=(1, 1, 1))
print(arima_forecast)
