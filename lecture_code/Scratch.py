from ARIMA import *
from Markowitz import *

input_dir = "../lecture_data"
file = "dataset.csv"
df = pd.read_csv(os.path.join(input_dir, file))
benchmark = pd.concat([df['Date'], df['KOSPI'], df['KOR10Y']], axis=1)
df.drop(columns=['KOSPI', 'KOR10Y'], inplace=True)
df_train_set = df.iloc[:-12, :]
df_test_set = df.iloc[-13:, :]

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

stock1 = True
if stock1:
    time_series = np.log(ARMA.data.iloc[:-12, 0] / ARMA.data.iloc[:-12, 0].shift(10)).dropna()
    ARMA.ACF_and_PACF_test(time_series)
    model_1 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(range(2, 3), 0, [2])).fit()
    model_2 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(range(1, 4), 0, 0)).fit()
    model_3 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(1, 0, 0)).fit()
    ARMA.evaluate_ARIMA(time_series, model_1, model_2, model_3)
    predicted1 = ARMA.forecasting(ARMA.data.iloc[:-12, 0], time_series, model_1, '2011-01-01', '2019-12-01')

stock2 = True
if stock2:
    time_series = np.log(ARMA.data.iloc[:-12, 1] / ARMA.data.iloc[:-12, 1].shift(10)).dropna()
    ARMA.ACF_and_PACF_test(time_series)
    model_1 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(range(2, 3), 0, [2])).fit()
    model_2 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(range(1, 4), 0, 0)).fit()
    model_3 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(1, 0, 0)).fit()
    ARMA.evaluate_ARIMA(time_series, model_1, model_2, model_3)
    predicted2 = ARMA.forecasting(ARMA.data.iloc[:-12, 1], time_series, model_1, '2011-01-01', '2019-12-01')

stock3 = True
if stock3:
    time_series = np.log(ARMA.data.iloc[:-12, 2] / ARMA.data.iloc[:-12, 2].shift(10)).dropna()
    ARMA.ACF_and_PACF_test(time_series)
    model_1 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(range(2, 3), 0, [2])).fit()
    model_2 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(range(1, 4), 0, 0)).fit()
    model_3 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(1, 0, 0)).fit()
    ARMA.evaluate_ARIMA(time_series, model_1, model_2, model_3)
    predicted3 = ARMA.forecasting(ARMA.data.iloc[:-12, 2], time_series, model_1, '2011-01-01', '2019-12-01')

stock4 = True
if stock4:
    time_series = np.log(ARMA.data.iloc[:-12, 3] / ARMA.data.iloc[:-12, 3].shift(10)).dropna()
    ARMA.ACF_and_PACF_test(time_series)
    model_1 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(range(2, 3), 0, [2])).fit()
    model_2 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(range(1, 4), 0, 0)).fit()
    model_3 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(1, 0, 0)).fit()
    ARMA.evaluate_ARIMA(time_series, model_1, model_2, model_3)
    predicted4 = ARMA.forecasting(ARMA.data.iloc[:-12, 3], time_series, model_1, '2011-01-01', '2019-12-01')

total = pd.concat([predicted1, predicted2, predicted3, predicted4], axis=1)
present_months = total.iloc[1:, :]
past_months = total.iloc[:-1, :]
past_months.index = present_months.index
total_mom_data = (present_months - past_months) / past_months

optimal = Optimization(df_train_set, benchmark)
optimal.guess_best_cov()
optimal.guess_best_mean_train_set()
optimal.optimalize_portfolio(optimal.cov, optimal.mean)
optimal.calculate_return_train_set(Plot=False)
optimal.profit.to_csv(os.path.join(input_dir, 'profit_train.csv'))
train_profit = optimal.profit

optimal = Optimization(df_test_set, benchmark)
profit_df = pd.DataFrame(index=optimal.mom_data.index, columns=optimal.mom_data.columns, dtype=float)
for i in range(len(df_test_set) - 1):
    optimal.guess_best_cov()
    optimal.guess_best_mean_test_set(i, predicted=total_mom_data)
    optimal.optimalize_portfolio(optimal.cov, optimal.mean)
    optimal.concat_return_test_set(i, profit_df)
optimal.calcualte_return_test_set(profit_df)
optimal.profit.to_csv(os.path.join(input_dir, 'profit_test.csv'))
test_profit = optimal.profit

profit = pd.concat([train_profit, test_profit])

optimal.plot_profit(profit)
