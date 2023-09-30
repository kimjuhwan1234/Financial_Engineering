import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt.solvers import qp
import matplotlib.pyplot as plt


class Optimization:
    def __init__(self, data: pd.DataFrame, mom_data: pd.DataFrame, benchmark: pd.DataFrame,
                 return_df: pd.DataFrame, var_df: pd.DataFrame, index_chunks: list):
        self.data = data
        self.mom_data = mom_data
        self.benchmark = benchmark
        self.return_df = return_df.astype(float)
        self.var_df = var_df.astype(float)
        self.index_chunks = index_chunks
        self.cov = []
        self.mean = []
        self.ini_mean = []
        self.sol = []
        self.profit = []

    def guess_initial_cov(self):
        Cov = matrix(self.mom_data.iloc[:12, :].cov().values)
        self.cov = Cov

    def guess_initial_mean(self):
        Mean = matrix(self.mom_data.iloc[:12, :].mean())
        self.mean = Mean

    def guess_test_cov(self, i: int):
        var = self.var_df.iloc[i, :]
        std = var.apply(np.sqrt)
        Cov = np.outer(std, std) * self.mom_data.iloc[self.index_chunks[i][0]:self.index_chunks[i][1], :].corr()
        Cov = matrix(Cov.values)
        self.cov = Cov

    def guess_test_mean(self, i: int):
        Mean = matrix(self.return_df.iloc[i, :])
        self.mean = Mean

    def optimalize_portfolio(self, Cov, Mean):
        n = 4
        r_min = 0.035

        G = matrix(np.concatenate((-np.transpose(Mean), -np.identity(n)), 0))
        h = matrix(np.concatenate((-np.ones((1, 1)) * r_min, np.zeros((n, 1))), 0))

        A = matrix(1.0, (1, n))
        b = matrix(1.0)
        q = matrix(np.zeros((n, 1)))
        sol = qp(Cov, q, G, h, A, b, options={'maxiters': 20})
        self.sol = list(sol['x'])

    def calculate_return(self, sol_df: pd.DataFrame, Plot: bool):
        profit_df = self.mom_data * sol_df
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
