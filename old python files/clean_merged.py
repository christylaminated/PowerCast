#check correlation with Germany/Luxembourg [€/MWh] Original resolutions
import pandas as pd

"""target_price = pd.read_csv("./expanded_hourly_data.csv", low_memory=False)
merged_df = pd.read_csv("./final_merged_dataset.csv", low_memory=False)
merged_df = merged_df.apply(pd.to_numeric, errors="coerce")  # Convert all columns to numeric
print("Columns with NaN values:\n", target_price.isna().sum())
correlation_matrix = merged_df.corr()
correlation_with_price = correlation_matrix["Germany/Luxembourg [€/MWh] Original resolutions"].abs().sort_values(ascending=False)
"""
#print(correlation_with_price.head(30), "were the most correlated features")  # Top 10 most correlated features

from sklearn.feature_selection import mutual_info_regression
merged_df = pd.read_csv("./final_merged_dataset.csv", low_memory=False)

print("Columns with NaN values AFTER filling:\n", merged_df.isna().sum())
# Find columns where all values are NaN
print("shape of merged_df",merged_df.shape)
columns_to_drop = [column for column in merged_df.columns if merged_df[column].isna().all()]

# Drop them
merged_df = merged_df.drop(columns=columns_to_drop)

# Print dropped columns
print("Dropped columns:", columns_to_drop)

#print(merged_df["Procurement price [€/MWh] Original resolutions"].isna().sum())
merged_df = merged_df.drop(["End date_x", "End date_y", 'timestamp'], axis=1)
#print((merged_df.map(lambda x: x.strip() if isinstance(x, str) else x) == "-").sum())
cols_to_drop = [col for col in merged_df.columns if (merged_df[col].astype(str) == "-").any()]
merged_df = merged_df.drop(columns=cols_to_drop)
print("Dropped columns pt2:", cols_to_drop)

#print("amount of - in your data after removal",(merged_df == '-').sum(), "and shape of merged_df is", merged_df.shape)
#print(merged_df.isna().sum(), "after attempt")
merged_df = merged_df.apply(lambda x: x.str.replace(",", "").astype(float) if x.dtype == "object" else x)

correlation_matrix = merged_df.corr()
corr_df = merged_df
correlation_with_price = correlation_matrix["Germany/Luxembourg [€/MWh] Original resolutions"].abs().sort_values(ascending=False)
print(correlation_with_price.head(40), "were the most correlated features") 

#y = target_price.drop(["timestamp"], axis=1)
y = merged_df["Germany/Luxembourg [€/MWh] Original resolutions"]
#X = merged_df.drop(["timestamp", "End date"], axis=1)
X = merged_df.drop(["Germany/Luxembourg [€/MWh] Original resolutions"], axis=1)


mi_scores = mutual_info_regression(X, y)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print(f"mi scores: {mi_series.head(40)}")

print(X.columns.tolist(), "shape: ", X.shape)
print(corr_df.columns.tolist(), "shape: ", corr_df.shape)








"""from sklearn.feature_selection import mutual_info_regression

X = merged_df.drop(columns=["Germany/Luxembourg (€/MWh)"])  # Features
y = merged_df["Germany/Luxembourg (€/MWh)"]  # Target variable

mi_scores = mutual_info_regression(X, y)  # Compute Mutual Information scores
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print(mi_series.head(10))  # Top 10 most important features
"""


"""
from scipy.stats import spearmanr

spearman_corrs = merged_df.corr(method="spearman")["Germany/Luxembourg (€/MWh)"].abs().sort_values(ascending=False)
print(spearman_corrs.head(10))
"""