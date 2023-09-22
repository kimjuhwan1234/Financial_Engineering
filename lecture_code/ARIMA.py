import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

warnings.filterwarnings("ignore")


class ARIMA:
    def __init__(self, data: pd.DataFrame):
        self.data = data

        present_months = self.data.iloc[1:, :]
        past_months = self.data.iloc[:-1, :]
        past_months.index = present_months.index
        self.mom_data = (present_months - past_months) / past_months

    def plot_stationary(self, data: pd.DataFrame, seasonality: int):
        for i in range(len(data.columns)):
            f, axes = plt.subplots(nrows=5, ncols=1, figsize=(9, 3 * 5))
            axes[0].plot(data.iloc[:, i], color='black', linewidth=1,
                         label=f'The original of {data.iloc[:, i].name}')
            axes[0].hlines(xmin=data.iloc[:, i].index[0], xmax=data.iloc[:, i].index[-1], y=0,
                           color='gray', linewidth=1)
            axes[0].legend()
            # 원본.

            axes[1].plot(data.iloc[:, i].diff(), color='black', linewidth=1,
                         label=f'First-difference of {data.iloc[:, i].name}')
            axes[1].hlines(xmin=data.iloc[:, i].index[0], xmax=data.iloc[:, i].index[-1], y=0,
                           color='gray', linewidth=1)
            axes[1].legend()
            # 1차 차분.

            axes[2].plot(data.iloc[:, i].diff().diff(), color='black', linewidth=1,
                         label=f'Second-difference of {data.iloc[:, i].name}')
            axes[2].hlines(xmin=data.iloc[:, i].index[0], xmax=data.iloc[:, i].index[-1], y=0,
                           color='gray', linewidth=1)
            axes[2].legend()
            # 2차 차분.

            axes[3].plot(np.log(data.iloc[:, i] / data.iloc[:, i].shift(seasonality)).dropna(),
                         color='black', linewidth=1,
                         label=f'No seasonal original of {data.iloc[:, i].name}')
            axes[3].hlines(xmin=data.iloc[:, i].index[0], xmax=data.iloc[:, i].index[-1], y=0,
                           color='gray', linewidth=1)
            axes[3].legend()
            # 계절성 제거 후 1차 차분.

            axes[4].plot(np.log(data.iloc[:, i] / data.iloc[:, i].shift(seasonality)).diff().dropna(),
                         color='black', linewidth=1,
                         label=f'No seasonal first-difference of {data.iloc[:, i].name}')
            axes[4].hlines(xmin=data.iloc[:, i].index[0], xmax=data.iloc[:, i].index[-1], y=0,
                           color='gray', linewidth=1)
            axes[4].legend()
            # 계절성 제거 후 1차 차분.

            plt.show()

    def adf_test(self, data: pd.DataFrame, seasonality: int):

        for i in range(len(data.columns)):
            if not seasonality:
                print(f'{data.iloc[:, i].name}')
                result = adfuller(data.iloc[:, i])
                print('Original')
                print(f'Statistics: {result[0]}')
                print(f'p-value: {result[1]}')
                print(f'Critical values: {result[4]}')
                print('---' * 40)

                result = adfuller(data.iloc[:, i].diff().dropna())
                print('The first difference')
                print(f'Statistics: {result[0]}')
                print(f'p-value: {result[1]}')
                print(f'Critical values: {result[4]}')
                print('---' * 40)

                result = adfuller(data.iloc[:, i].diff().diff().dropna())
                print('The second difference')
                print(f'Statistics: {result[0]}')
                print(f'p-value: {result[1]}')
                print(f'Critical values: {result[4]}')
                print('---' * 40)

            if seasonality:
                print(f'{data.iloc[:, i].name}')
                result = adfuller(np.log(data.iloc[:, i] / data.iloc[:, i].shift(seasonality)).diff().dropna())
                print('No seasonality of original')
                print(f'Statistics: {result[0]}')
                print(f'p-value: {result[1]}')
                print(f'Critical values: {result[4]}')
                print('---' * 40)

                result = adfuller(np.log(data.iloc[:, i] / data.iloc[:, i].shift(seasonality)).diff().diff().dropna())
                print('No seasonality of first difference')
                print(f'Statistics: {result[0]}')
                print(f'p-value: {result[1]}')
                print(f'Critical values: {result[4]}')
                print('---' * 40)

    def kpss_test(self, data: pd.DataFrame, seasonality: int):

        for i in range(len(data.columns)):
            if not seasonality:
                print(f'{data.iloc[:, i].name}')
                result = kpss(data.iloc[:, i])
                print('Original')
                print(f'Statistics: {result[0]}')
                print(f'p-value: {result[1]}')
                print(f'Critical values: {result[3]}')
                print('---' * 40)

                result = kpss(data.iloc[:, i].diff().dropna())
                print('The first difference')
                print(f'Statistics: {result[0]}')
                print(f'p-value: {result[1]}')
                print(f'Critical values: {result[3]}')
                print('---' * 40)

                result = kpss(data.iloc[:, i].diff().diff().dropna())
                print('The second difference')
                print(f'Statistics: {result[0]}')
                print(f'p-value: {result[1]}')
                print(f'Critical values: {result[3]}')
                print('---' * 40)

            if seasonality:
                print(f'{data.iloc[:, i].name}')
                result = kpss(np.log(data.iloc[:, i] / data.iloc[:, i].shift(seasonality)).diff().dropna())
                print('No seasonality of original')
                print(f'Statistics: {result[0]}')
                print(f'p-value: {result[1]}')
                print(f'Critical values: {result[3]}')
                print('---' * 40)

                result = kpss(np.log(data.iloc[:, i] / data.iloc[:, i].shift(seasonality)).diff().diff().dropna())
                print('No seasonality of first difference')
                print(f'Statistics: {result[0]}')
                print(f'p-value: {result[1]}')
                print(f'Critical values: {result[3]}')
                print('---' * 40)

    def ACF_and_PACF_test(self, data: pd.DataFrame):
        '''PACF의 첫째는 무시.
        ACF가 기하급수적으로 감소하지 않고 선형이면 MA doesn't exist.
        Negative value of ACF is not important.'''

        f, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 2 * 4))

        plot_acf(data, lags=20, ax=axes[0], title='Autocorrelations', color='black',
                 vlines_kwargs={'colors': 'black', 'linewidth': 5}, alpha=None)
        plot_pacf(data, lags=20, ax=axes[1], method='ols', title='PACF', color='gray',
                  vlines_kwargs={'colors': 'gray', 'linewidth': 5}, alpha=None)

        axes[1].hlines(xmin=0, xmax=20, y=2 * np.sqrt(1 / len(data)), label=f'{data.name}',
                       color='black', linewidth=1)
        axes[1].hlines(xmin=0, xmax=20, y=-2 * np.sqrt(1 / len(data)), color='black', linewidth=1)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def get_max_value(self, element):
        if isinstance(element, int) or isinstance(element, float):
            return element
        else:
            return max(element)

    def evaluate_ARIMA(self, time_series, model_1, model_2, model_3):
        summary_table = dict()
        for model in [model_1, model_2, model_3]:
            res = model
            temp_perf_dict = {}
            num_of_obs = len(time_series)

            q_statistics = res.test_serial_correlation(method='ljungbox', lags=12)[0]
            temp_perf_dict['SSE'] = round(res.sse, 3)
            temp_perf_dict['AIC'] = round(num_of_obs * np.log(res.sse) + 2 * len(res.params), 3)
            temp_perf_dict['SBC'] = round(num_of_obs * np.log(res.sse) + len(res.params) * np.log(num_of_obs), 3)
            temp_perf_dict['Q(1)'] = {'q_stats': round(q_statistics[0][0], 2), 'p_val': round(q_statistics[1][0], 3)}
            temp_perf_dict['Q(2)'] = {'q_stats': round(q_statistics[0][1], 2), 'p_val': round(q_statistics[1][1], 3)}
            temp_perf_dict['Q(3)'] = {'q_stats': round(q_statistics[0][2], 2), 'p_val': round(q_statistics[1][2], 3)}

            for param_name, param in zip(res.params.index, res.params):
                temp_perf_dict[param_name] = {'coef': round(param, 3), 't_stats': round(res.tvalues[param_name], 3)}
            hashable_order = tuple([tuple(order) if isinstance(order, list) == True else order for order in
                                    res.specification['order']])  # make res.specification['order'] hashable.
            hashable_s_order = tuple([tuple(s_order) if isinstance(s_order, list) == True else s_order for s_order in
                                      res.specification['seasonal_order']])  # make res.specification['order'] hashable.
            summary_table[(hashable_order, hashable_s_order)] = temp_perf_dict

            table_2_5 = pd.DataFrame()
            for key, value in summary_table.items():
                temp_series = pd.Series(value, name=key)
                table_2_5 = pd.concat([table_2_5, temp_series], axis=1)

        table_2_5.drop(index=['sigma2'], inplace=True)
        print(table_2_5.to_string())
        # 계절성이 있는 term을 추가 하면 다음과 같이 적는다. ex) 'ar.S.L1'/'ma.S.L12'

    def forecasting(self, data, time_series, model, start_date, predict_date):
        forecasts_m1 = model.forecast(steps=12)
        forecasts_m1.index = pd.date_range(start=time_series.index[-1] + pd.DateOffset(days=1), periods=12, freq='MS')
        full_seasonal_diff = pd.concat([time_series, forecasts_m1], axis=0)
        real_scale_forecasts = data.to_dict()
        indexer = full_seasonal_diff.index
        indexer = pd.to_datetime(indexer)

        for idx in np.where(indexer >= predict_date)[0]:
            temp_val = full_seasonal_diff[idx] + np.log(real_scale_forecasts[indexer[idx - 1]]) + np.log(
                real_scale_forecasts[indexer[idx - 12]]) - np.log(real_scale_forecasts[indexer[idx - 13]])
            real_scale_forecasts[indexer[idx]] = np.exp(temp_val)

        real_scale_forecasts_dataframe = pd.DataFrame.from_dict(real_scale_forecasts, orient='index')
        real_scale_forecasts_dataframe.columns = ['Ground Truth']
        # Figure for M1
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fitted = real_scale_forecasts_dataframe[
            (real_scale_forecasts_dataframe.index < predict_date) * (
                    real_scale_forecasts_dataframe.index >= start_date)]
        predicted = real_scale_forecasts_dataframe[real_scale_forecasts_dataframe.index >= predict_date]
        predicted.columns=[time_series.name]

        color = 'black'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('value', color=color)
        ax1.plot(fitted, color=color, linewidth=1, label='Ground Truth')
        ax1.plot(predicted, color='tab:blue', linewidth=1, linestyle='--', label='Model Forecast')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        plt.show()

        return predicted


if __name__ == "__main__":
    input_dir = "../lecture_data"
    file = "dataset.csv"
    df = pd.read_csv(os.path.join(input_dir, file))
    benchmark = pd.concat([df['Date'], df['KOSPI'], df['KOR10Y']], axis=1)
    df.drop(columns=['KOSPI', 'KOR10Y'], inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.iloc[:, :4]

    ARMA = ARIMA(df)
    # ARMA.plot_stationary(ARMA.mom_data, 10)
    # ARMA.adf_test(ARMA.mom_data, 10)
    # ARMA.kpss_test(ARMA.mom_data, 10)

    stock1 = True
    if stock1:
        time_series = np.log(ARMA.data.iloc[:-12, 0] / ARMA.data.iloc[:-12, 0].shift(10)).dropna()
        ARMA.ACF_and_PACF_test(time_series)
        model_1 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(range(2, 3), 0, [2])).fit()
        model_2 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(range(1, 4), 0, 0)).fit()
        model_3 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(1, 0, 0)).fit()
        ARMA.evaluate_ARIMA(time_series, model_1, model_2, model_3)
        predicted = ARMA.forecasting(ARMA.data.iloc[:-12, 0], time_series, model_1, '2011-01-01', '2019-12-01')
        print(predicted)
    stock2 = False
    if stock2:
        time_series = np.log(ARMA.data.iloc[:-12, 1] / ARMA.data.iloc[:-12, 1].shift(10)).dropna()
        ARMA.ACF_and_PACF_test(time_series)
        model_1 = sm.tsa.statespace.SARIMAX(time_series, trend='c', order=(range(2, 3), 0, [2])).fit()
        model_2 = sm.tsa.statespace.SARIMAX(time_series, trend='c', order=(range(1, 4), 0, 0)).fit()
        model_3 = sm.tsa.statespace.SARIMAX(time_series, trend='c', order=(1, 0, 0)).fit()
        ARMA.evaluate_ARIMA(time_series, model_1, model_2, model_3)
        ARMA.forecasting(ARMA.data.iloc[:-12, 1], time_series, '2011-01-01', '2019-12-01')

    stock3 = False
    if stock3:
        time_series = np.log(ARMA.data.iloc[:-12, 2] / ARMA.data.iloc[:-12, 2].shift(10)).dropna()
        ARMA.ACF_and_PACF_test(time_series)
        model_1 = sm.tsa.statespace.SARIMAX(time_series, trend='c', order=(range(2, 3), 0, [2])).fit()
        model_2 = sm.tsa.statespace.SARIMAX(time_series, trend='c', order=(range(1, 4), 0, 0)).fit()
        model_3 = sm.tsa.statespace.SARIMAX(time_series, trend='c', order=(1, 0, 0)).fit()
        ARMA.evaluate_ARIMA(time_series, model_1, model_2, model_3)
        ARMA.forecasting(ARMA.data.iloc[:-12, 2], time_series, '2011-01-01', '2019-12-01')

    stock4 = False
    if stock4:
        time_series = np.log(ARMA.data.iloc[:-12, 3] / ARMA.data.iloc[:-12, 3].shift(10)).dropna()
        ARMA.ACF_and_PACF_test(time_series)
        model_1 = sm.tsa.statespace.SARIMAX(time_series, trend='c', order=(range(2, 3), 0, [2])).fit()
        model_2 = sm.tsa.statespace.SARIMAX(time_series, trend='c', order=(range(1, 4), 0, 0)).fit()
        model_3 = sm.tsa.statespace.SARIMAX(time_series, trend='c', order=(1, 0, 0)).fit()
        ARMA.evaluate_ARIMA(time_series, model_1, model_2, model_3)
        ARMA.forecasting(ARMA.data.iloc[:-12, 3], time_series, '2011-01-01', '2019-12-01')
