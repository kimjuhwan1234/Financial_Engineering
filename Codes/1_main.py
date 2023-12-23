from ARIMA import *
from Markowitz import *

if __name__ == "__main__":
    if True:
        # Bring data & and I choose the number of firms to invest as twenty.
        n = 20
        input_dir = "../Database"
        file = "dataset.xlsx"
        df = pd.read_excel(os.path.join(input_dir, file))
        df.drop(index=0, inplace=True)
        df['Symbol'] = pd.to_datetime(df['Symbol'])
        df.set_index('Symbol', inplace=True)

        # Calculate the ground truth return for backtesting.
        present_months = df.iloc[1:, :]
        past_months = df.iloc[:-1, :]
        past_months.index = present_months.index
        mom_data = (present_months - past_months) / past_months
        mom_data.index = present_months.index
        mom_data = pd.DataFrame(mom_data)

        # Bring KOSPI data to compare with Fund-K.
        file = "Benchmark.csv"
        benchmark = pd.read_csv(os.path.join(input_dir, file))
        benchmark = benchmark.sort_values(by='Date', ascending=True)
        benchmark.set_index('Date', inplace=True)
        date = mom_data.index
        date = date[1:]
        benchmark.index = date
        benchmark = benchmark / benchmark.shift(1) - 1
        benchmark = benchmark.dropna()

        # Chunks periods as 10 years(120 months) for Rolling_Out_of_Sample Backtesting.
        total_index_length = len(mom_data)
        index_chunks = []
        for i in range(0, total_index_length):
            start_index = i
            if (start_index + 120) >= total_index_length:
                break
            index_chunks.append((start_index, start_index + 120))

        # First ten years are train set. So we need to save initial covariance, initial mean, and initial firm list.
        # I'll save conditional variances to var_df, which are predicted by GARCH.
        # I'll save predicted returns to return_df, which are predicted by ARIMA.
        ini_Cov = []
        ini_Mean = []
        ini_columns = []
        lst = [([1], 0, [1]) for _ in range(n)]
        var_df = pd.DataFrame(columns=mom_data.columns, index=mom_data.index[120:len(mom_data)])
        return_df = pd.DataFrame(columns=mom_data.columns, index=mom_data.index[120:len(mom_data)])

    Security_Selection = True
    if Security_Selection:
        # My security selection criteria is sharpe ratio.
        # I'll invest top 20 firms that have the highest sharpe ratio every backtesting.
        for i in range(len(index_chunks)):
            period = range(index_chunks[i][0], index_chunks[i][1])
            temporary_df = mom_data.iloc[period, :]
            sharpe_ratio = temporary_df.mean() / temporary_df.std()
            temporary_df = pd.concat([temporary_df, sharpe_ratio.to_frame().T], axis=0)
            sorted_df = temporary_df.T.sort_values(by=0, ascending=False).T
            top_20_df = sorted_df.iloc[:-1, :20]

            # First ten years is train set, so save historical cov, mean, and firm list.
            if i == 0:
                ini_Cov = matrix(top_20_df.cov().values)
                ini_Mean = matrix(top_20_df.mean())
                ini_columns = top_20_df.columns

            # Save predicted figures in var_df and return_df.
            ARMA = ARIMA(top_20_df)
            for j in range(n):
                r, var = ARMA.forecasting_ARMA_GARCH(j, lst[j])
                return_df.loc[return_df.index[i], top_20_df.columns[j]] = float(r)
                var_df.loc[var_df.index[i], top_20_df.columns[j]] = float(var)

        # I checked whether it goes successfully or not and no problem.
        var_df.to_csv('../Files/var_df.csv')
        return_df.to_csv('../Files/return_df.csv')

    portfolio = True
    if portfolio:
        # I'll save Markowitz model solution(no short selling) in sol_df.
        optimal = Optimization(mom_data, benchmark, return_df, var_df, index_chunks)
        sol_df = pd.DataFrame(columns=mom_data.columns, index=mom_data.index[120:len(mom_data)])
        sol_df = pd.DataFrame(index=optimal.mom_data.index, columns=optimal.mom_data.columns, dtype=float)

        # First 10 years solution should be same because it's train set.
        optimal.optimalize_portfolio(n, ini_Cov, ini_Mean)
        sol_df.loc[:sol_df.index[119], ini_columns] = optimal.sol

        # generating and save test set solution.
        for i in range(len(index_chunks)):
            optimal.guess_test_cov(i)
            optimal.guess_test_mean(i)
            optimal.optimalize_portfolio(n, optimal.cov, optimal.mean)
            indexes_with_values = return_df.columns[return_df.iloc[i].notna()]
            sol_df.loc[var_df.index[i], indexes_with_values] = optimal.sol

        # I checked whether it goes successfully or not and no problem.
        sol_df.to_csv('../Files/sol_df.csv')
        optimal.calculate_return(sol_df, Plot=True)
        optimal.profit.to_csv('../Files/profit_df.csv')

    portfolio_analysis = True
    if portfolio_analysis:
        # This part is calculating evaluation metric.
        result = pd.DataFrame(
            index=['Cumulative Return', 'monthly return min', 'monthly return max', 'annual return mean',
                   'annual return std', 'downside std'],
            columns=['Fund-K', 'KOSPI'])

        for i in range(2):
            row = optimal.profit.iloc[:, -i - 1]
            result.iloc[0, i] = float(row.sum())
            result.iloc[1, i] = np.min(row)
            result.iloc[2, i] = np.max(row)
            result.iloc[3, i] = np.exp(np.mean(row) * 12) - 1
            result.iloc[4, i] = np.exp(np.std(row) * np.sqrt(12)) - 1
            result.iloc[5, i] = np.exp(np.std(row[row < 0] * np.sqrt(12))) - 1

        row = optimal.profit.iloc[2:, -2:]
        MDD = pd.DataFrame(index=['Maximum drawdown'], columns=result.columns)
        row2 = np.exp(row.astype(float)) - 1
        cumulative_returns = np.cumprod(1 + row2) - 1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1)
        MDD.iloc[0, :] = drawdown.min()
        MDD[["Fund-K", "KOSPI"]] = MDD[["KOSPI", "Fund-K"]]
        result = pd.concat([result, MDD], axis=0)

        sharpe_ratio = pd.DataFrame(index=['Sharpe ratio'], columns=result.columns)
        sharpe_ratio.iloc[0, :] = result.iloc[3, :] / result.iloc[4, :]
        result = pd.concat([result, sharpe_ratio], axis=0)

        sortino_ratio = pd.DataFrame(index=['Sortino ratio'], columns=result.columns)
        sf = result.iloc[3, :] / result.iloc[5, :]
        sortino_ratio.iloc[0, :] = sf
        result = pd.concat([result, sortino_ratio], axis=0)

        calmar_ratio = pd.DataFrame(index=['Calmar ratio'], columns=result.columns)
        calmar = result.iloc[3, :] / abs(result.iloc[7, :])
        calmar_ratio.iloc[0, :] = calmar
        result = pd.concat([result, calmar_ratio], axis=0)

        result.to_csv('../Files/Evaluation_Metric.csv')
