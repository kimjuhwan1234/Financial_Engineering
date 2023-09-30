import os
import arch
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

        self.model = []

    def plot_time_series(self, data: pd.DataFrame, seasonality: int):
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

    def ACF_and_PACF_test(self, time_series):
        '''PACF의 첫째는 무시.
        ACF가 기하급수적으로 감소하지 않고 선형이면 MA doesn't exist.
        Negative value of ACF is not important.'''

        f, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 2 * 4))

        plot_acf(time_series, lags=20, ax=axes[0], title='Autocorrelations', color='black',
                 vlines_kwargs={'colors': 'black', 'linewidth': 5}, alpha=None)
        plot_pacf(time_series, lags=20, ax=axes[1], method='ols', title='PACF', color='gray',
                  vlines_kwargs={'colors': 'gray', 'linewidth': 5}, alpha=None)

        axes[1].hlines(xmin=0, xmax=20, y=2 * np.sqrt(1 / len(time_series)), label=f'{time_series.name}',
                       color='black', linewidth=1)
        axes[1].hlines(xmin=0, xmax=20, y=-2 * np.sqrt(1 / len(time_series)), color='black', linewidth=1)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def get_max_value(self, element):
        if isinstance(element, int) or isinstance(element, float):
            return element
        else:
            return max(element)

    def evaluate_ARIMA(self, time_series, lag_list: list):
        summary_table = dict()
        num_of_obs = len(time_series)
        table_2_5 = pd.DataFrame()

        for lag in lag_list:
            temp_perf_dict = {}
            max_values = [self.get_max_value(elem) for elem in lag]
            max_element = max(max_values)

            res = sm.tsa.statespace.SARIMAX(endog=time_series[12 - max_element:], order=lag, trend='n').fit()

            q_statistics = res.test_serial_correlation(method='ljungbox', lags=12)[0]
            temp_perf_dict['SSE'] = round(res.sse, 2)
            temp_perf_dict['AIC'] = round(num_of_obs * np.log(res.sse) + 2 * len(res.params), 2)
            temp_perf_dict['SBC'] = round(num_of_obs * np.log(res.sse) + len(res.params) * np.log(num_of_obs), 2)
            temp_perf_dict['Q(4)'] = {'q_stats': round(q_statistics[0][3], 2), 'p_val': round(q_statistics[1][3], 2)}
            temp_perf_dict['Q(8)'] = {'q_stats': round(q_statistics[0][7], 2), 'p_val': round(q_statistics[1][7], 2)}
            temp_perf_dict['Q(12)'] = {'q_stats': round(q_statistics[0][11], 2), 'p_val': round(q_statistics[1][11], 2)}

            for param_name, param in zip(res.params.index, res.params):
                temp_perf_dict[param_name] = {'coef': round(param, 2), 't_stats': round(res.tvalues[param_name], 2)}
            hashable_order = tuple(
                [tuple(order) if isinstance(order, list) == True else order for order in res.specification['order']])
            summary_table[hashable_order] = temp_perf_dict

        for key, value in summary_table.items():
            temp_series = pd.Series(value, name=key)
            table_2_5 = pd.concat([table_2_5, temp_series], axis=1)

        table_2_5.drop(index=['sigma2'], inplace=True)
        print(table_2_5.to_string())

    def plot_forecasting(self, time_series, lag, start_date, predict_date):
        model = sm.tsa.statespace.SARIMAX(endog=time_series, seasonal_order=lag, trend='n').fit()

        forecasts_m1 = model.forecast(steps=1)

        # Figure for M1
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fitted = time_series[(time_series.index < predict_date) * (time_series.index >= start_date)]

        color = 'black'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('value', color=color)
        ax1.plot(fitted, color=color, linewidth=1, label='Ground Truth')
        ax1.plot(forecasts_m1, color='tab:blue', linewidth=1, linestyle='--', label='Model Forecast')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        plt.show()

    def plot_forecasting_log(self, data, time_series, lag, start_date, predict_date):
        max_values = [self.get_max_value(elem) for elem in lag]
        max_element = max(max_values)
        model = sm.tsa.statespace.SARIMAX(endog=time_series[12 - max_element:], order=lag, trend='n').fit()

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
        predicted.columns = [time_series.name]

        color = 'black'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('value', color=color)
        ax1.plot(fitted, color=color, linewidth=1, label='Ground Truth')
        ax1.plot(predicted, color='tab:blue', linewidth=1, linestyle='--', label='Model Forecast')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        plt.show()

        self.model = model

        return predicted

    def estimate_forecasting_error(self, log: bool, time_series, lag, predict_date):
        checker = time_series.index <= predict_date

        train_set = time_series[checker]
        test_set = time_series[~checker]
        n_train = len(train_set)
        n_test = len(test_set)
        f1, f1_error = [], []
        ground_truth = []

        for i in range(n_test):
            crt_time = i + n_train
            x_train = time_series[:crt_time]

            if log:
                model_1 = sm.tsa.statespace.SARIMAX(endog=x_train, order=lag, trend='n').fit()

            if not log:
                model_1 = sm.tsa.statespace.SARIMAX(endog=x_train, seasonal_order=lag, trend='n').fit()

            # one-step-ahead forecasts
            forecast_1 = model_1.forecast(steps=1)

            # true one-step-ahead value
            y = time_series[crt_time]
            ground_truth.append(y)
            f1.append(forecast_1[0])
            f1_error.append(y - forecast_1[0])

        plt.figure(figsize=(12, 4))
        plt.plot(ground_truth, label='ground truth', color='k', linestyle='--')
        plt.plot(f1, label='f1 predicted', color='r')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.scatter(np.linspace(1, len(f1_error), len(f1_error)), f1_error, label='f1error', color='r')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.legend()
        plt.show()

        s_2000_3q = ground_truth[0]
        f1_2000_3q = f1[0]
        f1 = pd.Series(f1)
        f1_error = pd.Series(f1_error)

        print(f"Actual value:{round(s_2000_3q, 3)}, f1 forecast:{round(f1_2000_3q, 3)}")
        print(f"avg f1:{round(np.array(f1).mean(), 4)}")
        print(f"var of f1:{round(np.array(f1).var(), 4)}")
        print(f'mean squared prediction error of f1: {round((f1_error ** 2).mean(), 4)}')

    def forecasting_ARMA_GARCH(self, period: range, firm_number: int, lag: tuple):
        time_series = self.mom_data.iloc[period, firm_number]
        SARIMA_model = sm.tsa.statespace.SARIMAX(endog=time_series, seasonal_order=lag, trend='n').fit()
        GARCH_model = arch.arch_model(SARIMA_model.resid, vol='GARCH', p=1, q=1).fit(disp='off', show_warning=False)

        forecasts_m1 = SARIMA_model.forecast(steps=1)
        var = GARCH_model.conditional_volatility[-1]

        return forecasts_m1, var


if __name__ == "__main__":
    input_dir = "../lecture_data"
    file = "dataset.csv"
    df = pd.read_csv(os.path.join(input_dir, file))
    benchmark = pd.concat([df['Date'], df['KOSPI'], df['KOR10Y']], axis=1)
    df.drop(columns=['KOSPI', 'KOR10Y'], inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.iloc[:, :4]

    backtesting_chunk = True
    if backtesting_chunk:
        total_index_length = len(df)
        index_chunks = []

        for i in range(0, total_index_length):
            start_index = i
            if (start_index + 11) >= total_index_length:
                break
            index_chunks.append((start_index, start_index + 11))

        print(index_chunks)

    ARMA = ARIMA(df)

    overall = False
    if overall:
        ARMA.plot_time_series(ARMA.data, 10)
        ARMA.adf_test(ARMA.mom_data, 10)
        ARMA.kpss_test(ARMA.mom_data, 10)

    stock1 = False
    if stock1:
        test_time_series = np.log(ARMA.mom_data.iloc[:-12, 0] / ARMA.mom_data.iloc[:-12, 0].shift(10)).dropna()

        ARMA.ACF_and_PACF_test(test_time_series)

        lag_list = [([2], 0, [2]), (0, 0, [2]), ([2], 0, 0)]
        ARMA.evaluate_ARIMA(test_time_series, lag_list)

        time_series = ARMA.mom_data.iloc[:-12, 0]
        full_time_series = ARMA.mom_data.iloc[:, 0]

        ARMA.plot_forecasting(time_series, ([2], 0, [2], 10), '2011-01-01', '2019-12-01')

        ARMA.estimate_forecasting_error(False, full_time_series, ([2], 0, [2], 10), '2019-12-01')

    stock2 = False
    if stock2:
        test_time_series = np.log(ARMA.mom_data.iloc[:-12, 1] / ARMA.mom_data.iloc[:-12, 1].shift(10)).dropna()

        ARMA.ACF_and_PACF_test(test_time_series)

        lag_list = [([2], 0, [2]), (0, 0, [2]), ([2], 0, 0)]
        ARMA.evaluate_ARIMA(test_time_series, lag_list)

        time_series = ARMA.mom_data.iloc[:-12, 1]
        full_time_series = ARMA.mom_data.iloc[:, 1]

        ARMA.plot_forecasting(time_series, ([2], 0, [2], 10), '2011-01-01', '2019-12-01')

        ARMA.estimate_forecasting_error(False, full_time_series, ([2], 0, [2], 10), '2019-12-01')

    stock3 = False
    if stock3:
        test_time_series = np.log(ARMA.mom_data.iloc[:-12, 2] / ARMA.mom_data.iloc[:-12, 2].shift(10)).dropna()

        ARMA.ACF_and_PACF_test(test_time_series)

        lag_list = [([2], 0, [2]), (0, 0, [2]), ([2], 0, 0)]
        ARMA.evaluate_ARIMA(test_time_series, lag_list)

        time_series = ARMA.mom_data.iloc[:-12, 2]
        full_time_series = ARMA.mom_data.iloc[:, 2]

        ARMA.plot_forecasting(time_series, ([2], 0, [2], 10), '2011-01-01', '2019-12-01')

        ARMA.estimate_forecasting_error(False, full_time_series, ([2], 0, [2], 10), '2019-12-01')

    stock4 = False
    if stock4:
        test_time_series = np.log(ARMA.mom_data.iloc[:-12, 3] / ARMA.mom_data.iloc[:-12, 3].shift(10)).dropna()

        ARMA.ACF_and_PACF_test(test_time_series)

        lag_list = [([2], 0, [2]), (0, 0, [2]), ([2], 0, 0)]
        ARMA.evaluate_ARIMA(test_time_series, lag_list)

        time_series = ARMA.mom_data.iloc[:-12, 3]
        full_time_series = ARMA.mom_data.iloc[:, 3]

        ARMA.plot_forecasting(time_series, ([2], 0, [2], 10), '2011-01-01', '2019-12-01')

        ARMA.estimate_forecasting_error(False, full_time_series, ([2], 0, [2], 10), '2019-12-01')
