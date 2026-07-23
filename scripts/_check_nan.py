import pandas as pd
import numpy as np

path = r"\\10.10.10.1\NAS Processed Images\20260430_ZCH_BZ23_mut_1_with_marker_processed\readout\mapping_postcode.csv"
df = pd.read_csv(path)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print()
print("NaN counts per column:")
print(df.isna().sum())
print()
print("Dtypes:")
print(df.dtypes)
print()
nan_mask = df["Probability"].isna() | df["Entropy"].isna()
print(f"Rows with NaN in Probability or Entropy: {nan_mask.sum()}")
if nan_mask.any():
    print("\nSample NaN rows:")
    print(df[nan_mask].head(10).to_string())
    print(f"\nGene values for NaN rows:")
    print(df.loc[nan_mask, "Gene"].value_counts())
