import pandas as pd

# Load dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# First 5 rows
print("First 5 Rows")
print(df.head())

# Dataset shape
print("\nShape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Class distribution
print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

# Dataset information
print("\nDataset Info:")
print(df.info())