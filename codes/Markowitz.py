import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt.solvers import qp
import matplotlib.pyplot as plt


class Optimization:
    def __init__(self, mom_data: pd.DataFrame, benchmark: pd.DataFrame,
                 return_df: pd.DataFrame, var_df: pd.DataFrame, index_chunks: list):
        self.mom_data = mom_data
        self.benchmark = benchmark
        self.return_df = return_df.astype(float)
        self.var_df = var_df.astype(float)
        self.index_chunks = index_chunks
        self.cov = []
        self.mean = []
        self.sol = []
        self.profit = []

    def guess_test_cov(self, i: int):
        var = self.var_df.iloc[i, :].dropna().astype(float)
        std = var.apply(np.sqrt)
        indexes_with_values = self.var_df.columns[self.var_df.iloc[i].notna()]
        Cov = np.outer(std, std) * self.mom_data.loc[self.mom_data.index[self.index_chunks[i][0]]:self.mom_data.index[
            self.index_chunks[i][1]], indexes_with_values].corr()
        Cov = matrix(Cov.values)
        self.cov = Cov

    def guess_test_mean(self, i: int):
        indexes_with_values = self.return_df.columns[self.return_df.iloc[i].notna()]
        Mean = matrix(self.return_df.loc[self.return_df.index[i], indexes_with_values])
        self.mean = Mean

    def optimalize_portfolio(self, n, Cov, Mean):
        r_min = 0.01

        G = matrix(np.concatenate((-np.transpose(Mean), -np.identity(n)), 0))
        h = matrix(np.concatenate((-np.ones((1, 1)) * r_min, np.zeros((n, 1))), 0))

        A = matrix(1.0, (1, n))
        b = matrix(1.0)
        q = matrix(np.zeros((n, 1)))
        sol = qp(Cov, q, G, h, A, b)
        self.sol = list(sol['x'])

    def calculate_return(self, sol_df: pd.DataFrame, Plot: bool):
        profit_df = self.mom_data * sol_df
        profit_df = (profit_df + 1)
        profit_df = np.log(profit_df.astype(float))

        benchmark = (self.benchmark + 1)
        benchmark = np.log(benchmark.astype(float))

        profit_df['KOSPI'] = benchmark

        profit_df['Fund-K'] = profit_df.iloc[:, :-1].sum(axis=1)
        profit = profit_df
        self.profit = pd.DataFrame(profit)

        if Plot:
            result = profit_df.iloc[2:, -2:].cumsum(axis=0)
            plt.figure(figsize=(10, 6))
            color_dict = {'Fund-K': 'navy', 'KOSPI': 'gray'}
            labels = []
            handles = []

            for key in color_dict:
                if key in result.T.index:
                    idx = result.T.index.get_loc(key)
                    line, = plt.plot(result.index, result.T.iloc[idx].fillna(method='ffill'),
                                     label=key, color=color_dict[key])
                    handles.append(line)
                    labels.append(key)

            plt.title('RETURN')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Value')
            plt.xticks(rotation=45)
            plt.legend(handles=handles, labels=labels)
            plt.tight_layout()
            plt.show()
