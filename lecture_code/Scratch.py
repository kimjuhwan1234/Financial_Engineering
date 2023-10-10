from ARIMA import *
from Markowitz import *

if True:
    input_dir = "../lecture_data"
    file2 = "dataset.csv"
    df2 = pd.read_csv(os.path.join(input_dir, file2))
    benchmark = pd.concat([df2['Date'], df2['KOSPI'], df2['KOR10Y']], axis=1)
    df2.drop(columns=['KOSPI', 'KOR10Y'], inplace=True)

    file = "dataset.xlsx"
    df = pd.read_excel(os.path.join(input_dir, file))
    df.drop(index=0, inplace=True)
    df['Symbol'] = pd.to_datetime(df['Symbol'])
    df.set_index('Symbol', inplace=True)
    df = df.iloc[:, :4]

    present_months = df.iloc[1:, :]
    past_months = df.iloc[:-1, :]
    past_months.index = present_months.index
    mom_data = (present_months - past_months) / past_months
    mom_data.index = present_months.index
    mom_data = pd.DataFrame(mom_data)
    print(mom_data)

    file = "Benchmark.csv"
    df3 = pd.read_csv(os.path.join(input_dir, file))
    df3=df3.sort_values(by='Date', ascending=True)
    df3.set_index('Date', inplace=True)
    date = mom_data.index
    date = date[1:]
    df3.index = date
    df3=df3 / df3.shift(1) - 1
    df3=df3.dropna()

if True:
    total_index_length = len(df)
    index_chunks = []

    for i in range(0, total_index_length):
        start_index = i
        if (start_index + 12) >= total_index_length:
            break
        index_chunks.append((start_index, start_index + 12))

    print(index_chunks)

if True:
    ARMA = ARIMA(df)
    lst = [([1], 0, [1]), ([1], 0, [1]), ([1], 0, [1]), ([1], 0, [1])]

    var_df = pd.DataFrame(columns=df.columns[0:4], index=df.index[13:len(df)])
    return_df = pd.DataFrame(columns=df.columns[0:4], index=df.index[13:len(df)])

    for j in range(4):
        for i in range(len(index_chunks) - 1):
            period = range(index_chunks[i][0], index_chunks[i][1])
            r, var = ARMA.forecasting_ARMA_GARCH(period, j, lst[j])
            return_df.iloc[i, j] = float(r)
            var_df.iloc[i, j] = float(var)

portfolio = True
if portfolio:
    optimal = Optimization(df, mom_data, benchmark, return_df, var_df, index_chunks)
    sol_df = pd.DataFrame(index=optimal.mom_data.index, columns=optimal.mom_data.columns, dtype=float)

    optimal.guess_initial_cov()
    optimal.guess_initial_mean()
    optimal.optimalize_portfolio(optimal.cov, optimal.mean)
    sol_df.iloc[:12, :] = optimal.sol

    for i in range(len(index_chunks) - 1):
        print(i)
        optimal.guess_test_cov(i)
        optimal.guess_test_mean(i)
        optimal.optimalize_portfolio(optimal.cov, optimal.mean)
        sol_df.iloc[i + 12, :] = optimal.sol

    optimal.calculate_return(df3, sol_df, Plot=True)
    optimal.profit.to_csv(os.path.join(input_dir, 'profit_df.csv'))

portfolio_analysis = False
if portfolio_analysis:
    result = pd.DataFrame(
        index=['Cumulative Return', 'monthly return min', 'monthly return max', 'annual return mean',
               'annual return std', 'downside std'],
        columns=['Evaluation Metric(log scale)'])

    row = optimal.profit.iloc[:, 0]
    result.iloc[0, 0] = float(optimal.profit.sum())
    result.iloc[1, 0] = np.min(row)
    result.iloc[2, 0] = np.max(row)
    result.iloc[3, 0] = np.exp(np.mean(row) * 12) - 1
    result.iloc[4, 0] = np.exp(np.std(row) * np.sqrt(12)) - 1
    result.iloc[5, 0] = np.exp(np.std(row[row < 0] * np.sqrt(12))) - 1

    MDD = pd.DataFrame(index=['Maximum drawdown'], columns=result.columns)
    row = optimal.profit.iloc[:, 0]
    row2 = np.exp(row.astype(float)) - 1
    cumulative_returns = np.cumprod(1 + row2) - 1
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / (peak + 1)
    max_drawdown = drawdown.min()
    MDD.iloc[0, 0] = max_drawdown
    result = pd.concat([result, MDD], axis=0)

    sharpe_ratio = pd.DataFrame(index=['Sharpe ratio'], columns=result.columns)
    sharpe_ratio.iloc[0, 0] = result.iloc[3, 0] / result.iloc[4, 0]
    result = pd.concat([result, sharpe_ratio], axis=0)

    sortino_ratio = pd.DataFrame(index=['Sortino ratio'], columns=result.columns)
    sf = result.iloc[3, 0] / result.iloc[5, 0]
    sortino_ratio.iloc[0, 0] = sf
    result = pd.concat([result, sortino_ratio], axis=0)

    calmar_ratio = pd.DataFrame(index=['Calmar ratio'], columns=result.columns)
    calmar = result.iloc[3, 0] / abs(result.iloc[7, 0])
    calmar_ratio.iloc[0, 0] = calmar
    result = pd.concat([result, calmar_ratio], axis=0)

    result.to_csv('../lecture_data/Evaluation_Metric.csv')
