import os
import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt.solvers import qp
import matplotlib.pyplot as plt


class Optimization:
    def __init__(self, data: pd.DataFrame, benchmark: pd.DataFrame):
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        self.data = data.iloc[:, :4]

        present_months = self.data.iloc[1:, :]
        past_months = self.data.iloc[:-1, :]
        past_months.index = present_months.index
        self.mom_data = (present_months - past_months) / past_months
        self.mom_data.index = present_months.index

        self.benchmark = benchmark
        self.cov = []
        self.mean = []
        self.sol = []
        self.profit = []

    def guess_best_cov(self):
        Cov = matrix(self.mom_data.cov().values)
        self.cov = Cov

    def guess_best_mean_train_set(self):
        Mean = matrix(self.mom_data.mean())
        self.mean = Mean

    def guess_best_mean_test_set(self, i: int, predicted: pd.DataFrame):
        Mean = matrix(predicted.iloc[i, :])
        self.mean = Mean

    def optimalize_portfolio(self, Cov, Mean):
        n = 4
        r_min = 0.035

        G = matrix(np.concatenate((-np.transpose(Mean), -np.identity(n)), 0))
        h = matrix(np.concatenate((-np.ones((1, 1)) * r_min, np.zeros((n, 1))), 0))

        A = matrix(1.0, (1, n))
        b = matrix(1.0)
        q = matrix(np.zeros((n, 1)))
        sol = qp(Cov, q, G, h, A, b)

        self.sol = sol['x']

    def calculate_return_train_set(self, Plot: bool):
        sol = pd.DataFrame(self.sol).T
        profit_df = self.mom_data * sol.values

        profit_df = profit_df + 1
        profit_df = np.log(profit_df)

        profit_df['Sum'] = profit_df.sum(axis=1)
        profit = profit_df['Sum']
        self.profit = pd.DataFrame(profit)

        if Plot:
            result = profit.cumsum(axis=0)
            plt.figure(figsize=(10, 6))
            plt.plot(result)
            plt.title('return')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def concat_return_test_set(self, i: int, profit_df):
        sol = pd.DataFrame(self.sol).T
        moment = self.mom_data * sol.values
        profit_df.iloc[i, :] = moment.iloc[i, :]

    def calcualte_return_test_set(self, profit_df: pd.DataFrame):
        profit_df = profit_df + 1
        profit_df2 = pd.DataFrame(data=profit_df.values, columns=profit_df.columns, index=profit_df.index)
        profit_df = np.log(profit_df2)

        profit_df['Sum'] = profit_df.sum(axis=1)
        profit = profit_df['Sum']
        self.profit = pd.DataFrame(profit)

    def plot_profit(self, profit):
        result = profit.cumsum(axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(result)
        plt.title('return')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    input_dir = '../lecture_data'
    file = 'dataset.csv'
    df = pd.read_csv(os.path.join(input_dir, file))
    benchmark = pd.concat([df['Date'], df['KOSPI'], df['KOR10Y']], axis=1)
    df.drop(columns=['KOSPI', 'KOR10Y'], inplace=True)
    df_train_set = df.iloc[:-12, :]
    df_test_set = df.iloc[-13:, :]

    optimal = Optimization(df_train_set, benchmark)
    optimal.guess_best_cov()
    optimal.guess_best_mean_train_set()
    optimal.optimalize_portfolio(optimal.cov, optimal.mean)
    optimal.calculate_return_train_set(Plot=False)
    optimal.profit.to_csv(os.path.join(input_dir, 'profit_train.csv'))
    train_profit = optimal.profit

    if True:
        data = {
            'SAMSUNG ELECTRONICS': [
                0.217874, 0.647022, -0.220866, 0.173901, -0.669739, 0.206022,
                0.407026, 0.057400, -0.136006, -0.005381, 0.362327, 0.015247
            ],
            'NAVER': [
                0.061879, -0.363201, 0.454667, 0.759673, -0.787511, 0.011802,
                1.266847, 0.088440, -0.156750, 0.260578, 0.011495, 0.573589
            ],
            'LG CHEM': [
                -0.178525, 0.622124, -0.138101, -0.115059, 0.262219, -0.168990,
                -0.032508, -0.068940, -0.289151, 0.539944, -0.287735, 0.575324
            ],
            'HYUNDAI MOTORS': [
                -0.095824, 0.146936, -0.338883, 0.145771, 0.413995, 0.234764,
                0.058682, 0.117518, 0.289682, -0.175978, -0.098305, -0.062404
            ]
        }
        # 데이터프레임 생성
        predicted = pd.DataFrame(data)

        # 인덱스를 날짜 형식으로 생성
        predicted.index = pd.date_range(start='2020-01-01', periods=12, freq='MS')

    optimal = Optimization(df_test_set, benchmark)
    profit_df = pd.DataFrame(index=optimal.mom_data.index, columns=optimal.mom_data.columns, dtype=float)
    for i in range(len(df_test_set) - 1):
        optimal.guess_best_cov()
        optimal.guess_best_mean_test_set(i, predicted=predicted)
        optimal.optimalize_portfolio(optimal.cov, optimal.mean)
        optimal.concat_return_test_set(i, profit_df)
    optimal.calcualte_return_test_set(profit_df)
    optimal.profit.to_csv(os.path.join(input_dir, 'profit_test.csv'))
    test_profit = optimal.profit

    profit = pd.concat([train_profit, test_profit])

    optimal.plot_profit(profit)
