#final merged dataset

#actual consumption (Actual_consumption_Quarterhour.csv)
#Hydro pumped storage [MWh] Original resolutions   -0.423872
#Residual load [MWh] Original resolutions           0.244775

#actual generation (Actual_generation_Quarterhour.csv)
#Fossil gas [MWh] Original resolutions              0.409207
#Hydro pumped storage [MWh] Original resolutions    0.392169
#Other renewable [MWh] Original resolutions  2.265070 (MI)
#Nuclear [MWh] Original resolutions  0.556237 (MI)

#balancing energy- no linear (Balancing_energy_Quarterhour_Month.csv)
#Procurement price (-) [€/MW] Original resolutions  1.589477 (MI)

#frequency containment- no linear (Frequency_Containment_Reserve_202301010000_202503050000_Quarterhour.csv)
#Procurement price [€/MW] Original resolutions  1.497679 (MI)

#manual freqnency (Manual_Frequency_Restoration_Reserve_202301010000_202503050000_Quarterhour.csv)
#Volume procured (-) [MW] Original resolutions        0.146948
#Procurement price (+) [€/MW] Original resolutions    0.048239
#Procurement price (-) [€/MW] Original resolutions  1.390717 (MI)

#scheduled commercial- linear (Scheduled_commercial_exchanges_202301010000_202503050000_Quarterhour.csv)
#Luxembourg (export) [MWh] Original resolutions        0.608881
#Luxembourg (import) [MWh] Original resolutions        0.017487
#Netherlands (export) [MWh] Original resolutions      -0.010099
#MI for scheuld
#7         Denmark (import) [MWh] Original resolutions  8.086335
#     Luxembourg (export) [MWh] Original resolutions  7.999488
#    Netherlands (export) [MWh] Original resolutions  7.986774
#     Netherlands (import) [MWh] Original resolutions  7.963278
#         Poland (export) [MWh] Original resolutions  7.414996
#        Austria (export) [MWh] Original resolutions  7.291556
#         Denmark (export) [MWh] Original resolutions  6.87046

'''
1. merge all quarter hourly correctly
- change start date to timestamp
2. merge with output
3. make sure shape is correct
4. only train with certain column
'''
import os
import pandas as pd
import numpy as np

data_folder = "./"

csv_files = os.listdir(data_folder)

quarter_hourly_files = [file for file in csv_files if "Quarterhour" in file]

print(f"quarter hourly files: {quarter_hourly_files}")

dfs = [] #empty list to store dataframes

for file in quarter_hourly_files:
    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path, delimiter=";", low_memory=False)
    df.rename(columns={"Start date": "timestamp"}, inplace=True)
    df = df.astype(object)
    pd.set_option('future.no_silent_downcasting', True)
# Replace "-" with NaN
    df.replace("-", np.nan, inplace=True)
# Convert numeric columns back to appropriate types
    df = df.infer_objects(copy=False)   
    df.drop(columns=["End date"], inplace = True)
    #threshold = 10000  # Set a threshold for missing values
    df.dropna(axis=1, inplace=True)   
    df.drop_duplicates(subset="timestamp", inplace=True)
    dfs.append(df)
    print("shape of appended df: ", df.shape)

#print(f"Done looping over: {dfs}")
merged_df = dfs[0]
for i in range(1, len(dfs)):
    merged_df = pd.merge(merged_df, dfs[i], on="timestamp", how="inner")
    print("shape of merged_df at iteration ", i, " is ", merged_df.shape)


#print(f"head is {merged_df.head()}")
merged_df.to_csv("merged_quarter_hourly_data.csv", index=False)

#now we have correct merged df of quarterhourly to merge with our target output
target = pd.read_csv("Day-ahead_prices_202301010000_202503050000_Hour.csv", delimiter= ';', low_memory=False)
target.rename(columns={"Start date": "timestamp"}, inplace= True)
target.rename(columns={"Germany/Luxembourg [€/MWh] Original resolutions": "Germany Price"}, inplace = True)
target = target[["timestamp", "Germany Price"]]
target.drop_duplicates(subset="timestamp", inplace=True)
print("shape of target: ", target.shape, " and merged_df shape", merged_df.shape)
#Jan 1, 2023 12:00 AM
target["timestamp"] = pd.to_datetime(target["timestamp"], format= "%b %d, %Y %I:%M %p")
merged_df["timestamp"] = pd.to_datetime(target["timestamp"], format="%b %d, %Y %I:%M %p")

print("before expanding: ", target.shape)
target = target.loc[target.index.repeat(4)].reset_index(drop=True)

print("expanded: ", target.shape)

target["timestanp"] = pd.date_range(
    start = target["timestamp"].min(),
    periods=len(target),
    freq="15min"
)

final_merged = merged_df.merge(target, on="timestamp", how="inner")

print("final merged shape: ", final_merged.shape)

final_merged.to_csv("Linear_regression_data.csv", index=False)

final_merged.drop(columns=["timestamp"], inplace=True)
for col in final_merged.select_dtypes(include=["object"]).columns:
    final_merged[col] = final_merged[col].str.replace(",", "").astype(float)
correlation_matrix = final_merged.corr()  # Compute correlation for all columns
print(correlation_matrix.columns.tolist())  # Show all column names
target_corr = correlation_matrix["Germany Price"].sort_values(ascending=False)  # Replace with your column name
print(target_corr)





