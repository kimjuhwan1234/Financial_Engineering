from PCA_and_ETC import *

first_day_of_month = False
if first_day_of_month:
    # Creates a new table only containing the rows of dates that are first business day of the month
    dir = "../files/price_data/"
    df = pd.read_csv(dir + "adj_close.csv")

    months = df['Date'].apply(lambda x: x[5:7])
    df_filtered = df.loc[df.index[~(months == months.shift(1))], :]
    df_filtered.to_csv(dir + "first_day_of_month.csv", index=False)

log_momentum = False
if log_momentum:
    # For each month from 1990-01 to 2022-12, it creates a new table of 48 rows of momentum factor
    # Momentum Factor: ratio of the current month's value to the value from i months ago minus 1
    # mom_1 = r_{t-1}
    # mom_i = \prod_{j=t-i-1}^{t-2} (r_j+1) - 1, i \in 1,...,4

    df = pd.read_csv('../files/price_data/adj_close_first_day_of_month.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    print('[', end='')
    for current_date in df.index[50:]:
        window = df.loc[:current_date].tail(50)

        mom = pd.DataFrame(index=range(1, 50), columns=window.columns)

        log_mom_1 = np.log(window.iloc[-1]) - np.log(window.iloc[-2])
        mom.loc[1] = np.exp(log_mom_1) - 1
        for i in range(2, 50):
            log_mom_i = np.log(window.iloc[-2]) - np.log(window.shift(i - 1).iloc[-2])
            mom.loc[i] = np.exp(log_mom_i) - 1

        mom = mom.T

        # Delete rows with all NaN values
        mom = mom.dropna(how='any')

        filename = current_date.strftime('%Y-%m') + '.csv'
        mom.to_csv('../files/characteristics/' + filename, index_label='Momentum Index')

        print('-', end='')
        if int(current_date.strftime('%m')) == 12:
            print(f'] {current_date.strftime("%Y")} done!\n[')
    if int(current_date.strftime('%m')) != 12:
        print(f'] {current_date.strftime("%Y-%m")} done!\n[')

momentum = False
if momentum:
    # For each month from 1990-01 to 2022-12, it creates a new table of 48 rows of momentum factor
    # Momentum Factor: ratio of the current month's value to the value from i months ago minus 1
    # mom_1 = r_{t-1}
    # mom_i = \prod_{j=t-i-1}^{t-2} (r_j+1) - 1, i \in 1,...,4

    df = pd.read_csv('../files/price_data/adj_close_first_day_of_month.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    print('[', end='')
    for current_date in df.index[50:]:
        window = df.loc[:current_date].tail(50)

        mom = pd.DataFrame(index=range(1, 50), columns=window.columns)

        mom.loc[1] = window.iloc[-1] / window.iloc[-2] - 1
        for i in range(2, 50):
            mom.loc[i] = window.iloc[-2] / window.shift(i - 1).iloc[-2] - 1

        mom = mom.T

        # Delete rows with all NaN values
        mom = mom.dropna(how='any')

        filename = current_date.strftime('%Y-%m') + '.csv'
        mom.to_csv('../files/characteristics/' + filename, index_label='Momentum Index')

        print('-', end='')
        if int(current_date.strftime('%m')) == 12:
            print(f'] {current_date.strftime("%Y")} done!\n[')
        if int(current_date.strftime('%m')) != 12:
            print(f'] {current_date.strftime("%Y-%m")} done!\n[')

MOM_Merge = True
if MOM_Merge:
    directory = '../files/characteristics'
    long_short = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

    merged_df = pd.DataFrame()

    for file in long_short:
        data = pd.read_csv(os.path.join(directory, file))

        # Keep only the column that contains tickers and '1' columns
        data = pd.concat([data.iloc[:, 0], data.loc[:, momentum_prefix_finder(data) + '1']], axis=1)

        file_column_name = os.path.splitext(file)[0]

        # Rename the columns
        data = data.rename(columns={data.columns[0]: 'Firm Name', data.columns[1]: file_column_name})

        if merged_df.empty:
            merged_df = data
        else:
            merged_df = pd.merge(merged_df, data, on='Firm Name', how='outer')

    merged_df = merged_df.sort_values('Firm Name')

# fixing abnormal mom1
#     for col in merged_df.columns:
#         if col == 'Firm Name':
#             continue
#         merged_df.loc[merged_df[col] > 1, col] = 1
#         merged_df.loc[merged_df[col] < -0.5, col] = -0.5

    merged_df.to_csv('../files/mom1_data_combined_adj_close.csv', index=False)
