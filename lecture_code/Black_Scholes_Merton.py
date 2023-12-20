'''
Let S0 is the current price of the underlying asset;
K is the strike price;
r is the risk-free rate;
σ is the volatility of the underlying asset;
and ∆t = T /N where T is the time to maturity and N is the number of periods in the model.

Show (1) the corresponding binomial tree (in forms of matrix)
and (2) the price of the following options based on the binomial option pricing model.

1. European call option (S0 = 70, K = 60, T = 10, r = 0.05, σ = 0.2, N = 10)
2. American call option (S0 = 70, K = 60, T = 10, r = 0.05, σ = 0.2, N = 10)
3. European put option (S0 = 100, K = 95, T = 5, r = 0.04, σ = 0.1, N = 10)
4. American put option (S0 = 100, K = 95, T = 5, r = 0.04, σ = 0.1, N = 10)'''

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class option:
    def __init__(self, S, K, T, r, sigma, N, european=bool, call=bool):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.option = european
        self.option_class = call

    def delta(self):
        return self.T / self.N

    def R(self):
        return np.exp(self.r * self.delta())

    def udq(self):
        u = np.exp(self.sigma * np.sqrt(self.delta()))
        d = 1 / u
        q = (self.R() - d) / (u - d)
        return u, d, q

    def calculate_binomial_tree(self):
        matrix = pd.DataFrame(columns=range(self.N), index=range(self.N))
        u, d, q = self.udq()

        for i in range(self.N):
            for j in range(0, i + 1):
                matrix.iloc[j, i] = self.S * (d ** j) * (u ** (i - j))

        return matrix

    def calculate_max(self):
        matrix = self.calculate_binomial_tree()
        if self.option_class == True:
            matrix = matrix - self.K
            matrix[matrix < 0] = 0

        else:
            matrix = self.K - matrix
            matrix[matrix < 0] = 0

        return matrix

    def calculate_price(self):
        matrix = self.calculate_max()
        u, d, q = self.udq()
        # European
        if self.option == True:
            for i in range(self.N-1):
                for j in range(self.N-1-i):
                    matrix.iloc[j, -2 - i] = (q * matrix.iloc[j, -1 - i] + (1 - q) * matrix.iloc[
                        j + 1, -1 - i]) / self.R()

            print(matrix.iloc[0,0])

        # American
        else:
            matrix2 = self.calculate_max()
            for i in range(self.N - 1):
                for j in range(self.N - 1 - i):
                    matrix.iloc[j, -2 - i] = (q * matrix.iloc[j, -1 - i] + (1 - q) * matrix.iloc[
                        j + 1, -1 - i]) / self.R()

                    if matrix2.iloc[j,-2-i] > matrix.iloc[j, -2-i]:
                        matrix.iloc[j, -2 - i] = matrix2.iloc[j,-2-i]

            print(matrix.iloc[0,0])


if __name__ == "__main__":
    op = option(70, 60, 10, 0.05, 0.2, 10, True, True)
    print(op.calculate_binomial_tree())
    op.calculate_price()
    op = option(70, 60, 10, 0.05, 0.2, 10, False, True)
    print(op.calculate_binomial_tree())
    op.calculate_price()
    op = option(100, 95, 5, 0.04, 0.1, 10, True, False)
    print(op.calculate_binomial_tree())
    op.calculate_price()
    op = option(100, 95, 5, 0.04, 0.1, 10, False, False)
    print(op.calculate_binomial_tree())
    op.calculate_price()
