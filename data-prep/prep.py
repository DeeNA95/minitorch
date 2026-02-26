from datetime import datetime

import numpy as np
import pandas as pd

df = pd.read_csv("data/sales.tsv", sep="\t", header=None)
df = df.dropna()
print(df.head())
df["date"] = pd.to_datetime(df[2])
df["product_encoded"], ne = pd.factorize(df[0])
print(ne)
df = pd.get_dummies(df, columns=[6], dtype=int)
# df["category_encoded"], nen = pd.factorize(df[6])
df = df.drop([0, 2, 1, 3, 7], axis=1)
# print(nen)
print(df.columns)


def doy(time):
    t = time.strftime("%j")
    return int(t)


vect_doy = np.vectorize(doy)
df = df.to_numpy()
doys = vect_doy(df[:, 4])
print(doys)

df = np.concatenate((df, doys.reshape(-1, 1)), axis=1)
print(df[0])
indices = np.where(df[:, 3] == 3)


new_df = np.delete(df, indices[0], 0)

new_df = np.delete(new_df, [3, 4, 5], axis=1)  # labels = np.unique(new_df)
# drop product encode and normalise
new_df[:, -1] = new_df[:, -1] / 365
new_df[:, 2] = new_df[:, 2] / new_df[:, 2].max()
new_df[:, 0] = new_df[:, 0] / new_df[:, 0].max()
new_df[:, 1] = new_df[:, 1] / new_df[:, 1].max()
print(new_df[675])


print(new_df.shape)

y = new_df[:, 0].astype(np.float32)  # Quantity (Target)
X = new_df[:, 1:].astype(np.float32)  # Everything Else (Inputs)
# Save to minitorch/data directory
X.tofile("data/X_sales.bin")
y.tofile("data/y_sales.bin")
print("X shape:", X.shape)  # Should be (366497, 18)
print("y shape:", y.shape)  # Should be (366497,)
