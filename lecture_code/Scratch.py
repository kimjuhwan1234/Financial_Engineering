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
    print(df)

    present_months = df.iloc[1:, :]
    past_months = df.iloc[:-1, :]
    past_months.index = present_months.index
    mom_data = (present_months - past_months) / past_months
    mom_data.index = present_months.index
    mom_data = pd.DataFrame(mom_data)
    print(mom_data)

if True:
    total_index_length = len(df)
    index_chunks = []

    for i in range(0, total_index_length):
        start_index = i
        if (start_index + 11) >= total_index_length:
            break
        index_chunks.append((start_index, start_index + 11))

    print(index_chunks)

if True:
    ARMA = ARIMA(df)
    lst = [([2], 0, [2], 10), ([2], 0, [2], 10), ([2], 0, [2], 10), ([2], 0, [2], 10)]

    var_df = pd.DataFrame(columns=df.columns[0:4], index=df.index[12:120])
    return_df = pd.DataFrame(columns=df.columns[0:4], index=df.index[12:120])

    for j in range(4):
        for i in range(len(index_chunks) - 1):
            period = range(index_chunks[i][0], index_chunks[i][1])
            r, var = ARMA.forecasting(period, j, lst[j])
            return_df.iloc[i, j] = r
            var_df.iloc[i, j] = var

portfolio = True
if portfolio:
    return_df=return_df.astype(float)
    var_df=var_df.astype(float)
    return_df.to_csv('../lecture_data/return_df_before.csv')
    var_df.to_csv('../lecture_data/var_df_before.csv')

    optimal = Optimization(df, mom_data, benchmark, return_df, var_df, index_chunks)
    sol_df = pd.DataFrame(index=optimal.mom_data.index, columns=optimal.mom_data.columns, dtype=float)

    optimal.guess_initial_cov()
    optimal.guess_initial_mean()
    optimal.detect_error()

    optimal.return_df.to_csv('../lecture_data/return_df_after.csv')
    optimal.var_df.to_csv('../lecture_data/var_df_after.csv')

    optimal.optimalize_portfolio(optimal.cov, optimal.mean)

    sol_df.iloc[:12, :] = optimal.sol

    for i in range(len(index_chunks) - 1):
        print(i)
        optimal.guess_test_cov(i)
        optimal.guess_test_mean(i)
        optimal.optimalize_portfolio(optimal.cov, optimal.mean)
        sol_df.iloc[i + 12, :] = optimal.sol

    optimal.calculate_return(sol_df, Plot=True)
    optimal.profit.to_csv(os.path.join(input_dir, 'profit_df.csv'))
