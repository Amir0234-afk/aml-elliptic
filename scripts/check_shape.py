import pandas as pd

# replace with the real path to your raw features file
path = "data/raw/elliptic_txs_features.csv"

# read without interpreting header row
df = pd.read_csv(path, header=None)

print("shape:", df.shape)
print("first row (column count):", len(df.columns))
print("first row values part:", df.iloc[0].tolist()[:10])
