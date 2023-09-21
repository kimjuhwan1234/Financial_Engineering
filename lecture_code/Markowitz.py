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

        self.benchmark = benchmark
        self.cov = []
        self.mean = []
        self.sol = []
        self.profit = []

    def guess_best_cov(self):
        Cov = matrix(self.mom_data.cov().values)
        self.cov = Cov

    def guess_best_mean(self):
        Mean = matrix(self.mom_data.mean())
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

    def calculate_return(self, Plot: bool):
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


if __name__ == "__main__":
    input_dir = '../lecture_data'
    file = 'dataset.csv'
    df = pd.read_csv(os.path.join(input_dir, file))
    benchmark = pd.concat([df['Date'], df['KOSPI'], df['KOR10Y']], axis=1)
    df.drop(columns=['KOSPI', 'KOR10Y'], inplace=True)

    optimal = Optimization(df, benchmark)
    optimal.guess_best_cov()
    optimal.guess_best_mean()
    optimal.optimalize_portfolio(optimal.cov, optimal.mean)
    optimal.calculate_return(Plot=False)
    optimal.profit.to_csv(os.path.join(input_dir, 'profit.csv'))
