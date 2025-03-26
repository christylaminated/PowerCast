import pandas as pd
import numpy as np
target = pd.read_csv("Day-ahead_prices_202301010000_202503050000_Hour.csv", delimiter= ';', low_memory=False)
df = pd.read_csv("Scheduled_commercial_exchanges_202301010000_202503050000_Quarterhour.csv", delimiter= ';', low_memory=False)


#need to extend target

df.rename(columns={"Start date":"timestamp"}, inplace="True")
df["timestamp"] = pd.to_datetime(df["timestamp"], format = "%b %d, %Y %I:%M %p")
df = df.drop(["End date"], axis=1)
print("comparing with shape: ",df.shape)

target.rename(columns={"Start date":"timestamp"}, inplace="True")
target["timestamp"] = pd.to_datetime(df["timestamp"], format = "%b %d, %Y %I:%M %p")

print("before expanding: ", target.shape, target.head(3))
target_expanded = target.loc[target.index.repeat(4)].reset_index(drop=True)
print("expanded: ", target_expanded.shape, target_expanded.head(3))

target_expanded["timestamp"] = pd.date_range(
    start=df["timestamp"].min(), 
    periods=len(target_expanded), 
    freq="15min"
)



print("comparing with shape: ",df.shape)
target_expanded = target_expanded[["Germany/Luxembourg [€/MWh] Original resolutions", "timestamp"]]

merged_df = target_expanded.merge(df, on="timestamp", how="inner")

merged_df = merged_df.drop(["timestamp"], axis=1)

print("merged df after drop: ", merged_df.columns.tolist())

# Count occurrences of "-"
'''merged_df.replace("-", 0, inplace=True)
num_hyphen_values = (merged_df == "-").sum().sum()'''
merged_df.replace("-", np.nan, inplace=True)

# Step 1: Forward fill (use last known value)
merged_df.ffill(inplace=True)

# Step 2: Identify remaining NaNs that need to be filled with mean of top & bottom
for col in merged_df.columns:
    nan_indices = merged_df[col][merged_df[col].isna()].index  # Find NaN positions
    
    for idx in nan_indices:
        # Get values above and below
        top_value = merged_df[col][idx - 1] if idx > 0 else np.nan
        bottom_value = merged_df[col][idx + 1] if idx < len(merged_df) - 1 else np.nan
        
        # Fill NaN with mean if both top & bottom exist
        if not pd.isna(top_value) and not pd.isna(bottom_value):
            merged_df.at[idx, col] = (top_value + bottom_value) / 2

# Step 3: Backward fill to handle remaining NaNs
merged_df.bfill(inplace=True)

num_hyphen_values = (merged_df == "-").sum().sum()
nan_cols = merged_df.columns[merged_df.isna().any()].tolist()
if nan_cols:
    print("Columns with NaNs:", nan_cols)

print(f"Total occurrences of '-': {num_hyphen_values}")
merged_df = merged_df.apply(pd.to_numeric, errors="coerce")


correlation_matrix = merged_df.corr()  # Compute correlation for all columns
target_corr = correlation_matrix["Germany/Luxembourg [€/MWh] Original resolutions"].sort_values(ascending=False)  # Replace with your column name

print(target_corr)

from sklearn.feature_selection import mutual_info_regression

print("Does 'Germany/Luxembourg [€/MWh] Original resolutions' exist?", "Germany/Luxembourg [€/MWh] Original resolutions" in merged_df.columns)
print("NaN count in y:", merged_df.isna().sum())  # Should be 0
print("Data type of y:", merged_df.dtypes)  # Should be numeric
print("Shape is", merged_df.shape)

#merged_df["Hydro pumped storage [MWh] Original resolutions"].fillna(method="ffill", inplace=True)
#merged_df["Hydro pumped storage [MWh] Original resolutions"].fillna(method="bfill", inplace=True)
#print("NaN count in y:", merged_df["Hydro pumped storage [MWh] Original resolutions"].isna().sum())

#for actual generatoin
#X = merged_df[["Hydropower [MWh] Original resolutions", "Other renewable [MWh] Original resolutions", "Nuclear [MWh] Original resolutions", "Other conventional [MWh] Original resolutions"]]
nan_threshold = 50000  # Set threshold

# Identify columns to drop
columns_to_drop = merged_df.columns[merged_df.isna().sum() > nan_threshold]

# Drop those columns
merged_df.drop(columns=columns_to_drop, inplace=True)

X = merged_df.fillna(method="ffill").fillna(method="bfill")  # Forward + Backward Fill
y = merged_df["Germany/Luxembourg [€/MWh] Original resolutions"]
mi_scores = mutual_info_regression(X, y)
# Convert to Pandas DataFrame for sorting
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})

# Sort the DataFrame by MI Score
mi_df_sorted = mi_df.sort_values(by="MI Score", ascending=False)

print(mi_df_sorted)
print("shape after merging: ", merged_df.shape, merged_df.columns.tolist())