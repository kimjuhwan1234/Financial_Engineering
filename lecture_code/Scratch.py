import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../lecture_data/var_df.csv')
df = df.iloc[132, 1:]
df = df.astype(int)
df = df[df != 0].dropna()
print(df)

plt.figure(figsize=(10, 6))
plt.hist(df, bins=134, color='navy', edgecolor='black')
plt.title('Histogram of stock traded')
plt.xlabel('Number of investments')
plt.ylabel('Number of Firms')
plt.show()
