from ARIMA import *
from Markowitz import *

if True:
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
    predicted1 = ARMA.forecasting(ARMA.data.iloc[:-12, 0], time_series, model_1, '2011-01-01', '2019-12-01')

stock2 = True
if stock2:
    time_series = np.log(ARMA.data.iloc[:-12, 1] / ARMA.data.iloc[:-12, 1].shift(10)).dropna()
    ARMA.ACF_and_PACF_test(time_series)
    model_1 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(5, 0, 0)).fit()
    model_2 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=([1, 3, 5], 0, 0)).fit()
    model_3 = sm.tsa.statespace.SARIMAX(time_series, trend='n', order=(5, 0, 5)).fit()
    ARMA.evaluate_ARIMA(time_series, model_1, model_2, model_3)
    predicted2 = ARMA.forecasting(ARMA.data.iloc[:-12, 1], time_series, model_2, '2011-01-01', '2019-12-01')

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
    predicted4 = ARMA.forecasting(ARMA.data.iloc[:-12, 3], time_series, model_3, '2011-01-01', '2019-12-01')

portfolio = True
if portfolio:
    total = pd.concat([predicted1, predicted2, predicted3, predicted4], axis=1)
    present_months = total.iloc[1:, :]
    past_months = total.iloc[:-1, :]
    past_months.index = present_months.index
    total_mom_data = (present_months - past_months) / past_months

    # total_mom_data에서 전부 음수인 행 찾기
    negative_rows = total_mom_data[(total_mom_data < 0).all(axis=1)]

    # 절댓값이 가장 작은 음수를 양수로 바꾸기
    for index, row in negative_rows.iterrows():
        min_negative_value = row.min()
        min_negative_index = row.idxmin()  # 가장 작은 음수 값의 열 인덱스 찾기
        total_mom_data.at[index, min_negative_index] = -min_negative_value

    # 결과 출력
    print(total_mom_data)

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
        print(i)
        optimal.guess_best_cov()
        optimal.guess_best_mean_test_set(i, predicted=total_mom_data)
        optimal.optimalize_portfolio(optimal.cov, optimal.mean)
        optimal.concat_return_test_set(i, profit_df)

    optimal.calcualte_return_test_set(profit_df)
    optimal.profit.to_csv(os.path.join(input_dir, 'profit_test.csv'))
    test_profit = optimal.profit

    profit = pd.concat([train_profit, test_profit])
    optimal.plot_profit(profit)
