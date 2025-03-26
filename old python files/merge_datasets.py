import os
import pandas as pd
import numpy as np

data_folder = "./"

csv_files = os.listdir(data_folder)
#print("found csv files:", csv_files)

quarter_hourly_files = [file for file in csv_files if "Quarterhour.csv" in file]

print(f"quarter hourly files: {quarter_hourly_files}")

dfs = [] #empty list to store dataframes

for file in quarter_hourly_files:
    file_path = os.path.join(data_folder, file)
    #print(f"file_path: {file_path}")
    df = pd.read_csv(file_path, delimiter=";", low_memory=False)
    #print(f"head: {df.head()}")
    df["timestamp"] = pd.to_datetime(df["Start date"], format="%b %d, %Y %I:%M %p")
    df = df.drop(columns=["Start date", "End date"], errors="ignore")
    dfs.append(df)

print(f"Done looping over: {dfs}")
merged_df = dfs[0]
for i in range(1, len(dfs)):
    merged_df = pd.merge(merged_df, dfs[i], on="timestamp", how="inner")


print(f"head is {merged_df.head()}")
merged_df.to_csv("merged_quarter_hourly_data.csv", index=False)



#expand hourly data to quarter hourly data
hourly_files = [file for file in csv_files if "Hour.csv" in file]
print(f"hourly files found: {hourly_files}")
#there is only one hourly file so:
hourly_file = hourly_files[0]
file_path = os.path.join(data_folder, hourly_file)
df_hourly = pd.read_csv(file_path, delimiter=";", low_memory=False)
#rename start date to timestamp, inplace=True -> replaces it, no need to assign it back
df_hourly.rename(columns={"Start date":"timestamp"}, inplace="True")
df_hourly["timestamp"] = pd.to_datetime(df_hourly["timestamp"], format="%b %d, %Y %I:%M %p")
df_hourly.drop(columns=["End date"], inplace=True, errors="ignore")

print(f"before resampling: {df_hourly.shape}")

# Remove duplicate timestamps before resampling
#df_hourly = df_hourly.drop_duplicates(subset=["timestamp"])
df_hourly = df_hourly.drop_duplicates(subset=["timestamp"], keep="first")

#set index for resampling
df_hourly.set_index("timestamp", inplace=True)
#expand hourly data to 15Time intervals, forward fill, convert index back to column
df_hourly = df_hourly.resample("15min").ffill().reset_index()
print(f"after resampling: {df_hourly.shape}")
df_hourly = df_hourly[["timestamp", "Germany/Luxembourg [€/MWh] Original resolutions"]]
df_hourly.to_csv("expanded_hourly_data.csv", index = False)




#expand monthly data to quarter hourly data
monthly_files = [file for file in csv_files if "Month.csv" in file and not "Quarterhour" in file]
print(f"hourly files found: {monthly_files}")
file_path = os.path.join(data_folder, monthly_files[0])
df_monthly = pd.read_csv(file_path, delimiter=";", low_memory=False)

#rename it to timestamp
df_monthly.rename(columns={"Start date":"timestamp"}, inplace=True, errors="ignore")
df_monthly["timestamp"] = pd.to_datetime(df_monthly["timestamp"], format = "%b %d, %Y")

#generate/expand data to quarter hourly timestamp range
start_date = df_monthly["timestamp"].min()
end_date = df_monthly["timestamp"].max() + pd.DateOffset(months=1)
full_time_range = pd.date_range(start=start_date, end = end_date, freq="15min")[:-1]

#empty dataframe with full timestamp index
df_expanded = pd.DataFrame(index=full_time_range)
df_expanded.index.name = "timestamp"

#merge with monthly data
df_monthly.set_index("timestamp", inplace=True)
df_expanded = df_expanded.merge(df_monthly, how="left", left_index=True, right_index=True)
df_expanded.ffill(inplace=True) #fill each month's value across all timestamps

#reset index to bring timestamp back as a column
df_expanded.reset_index(inplace=True)
print(f"dfmonthly: {df_monthly.shape} and dfexpanded: {df_expanded.shape}")
df_expanded.to_csv("expanded_montly_data.csv", index=False)







#expand yearly data to quarter hourly data
yearly_files = [file for file in csv_files if "Year.csv" in file]
file_path = os.path.join(data_folder, yearly_files[0])
df_yearly = pd.read_csv(file_path, delimiter=";", low_memory=False)

#rename to timestamp
df_yearly.rename(columns={"Start date": "timestamp"}, inplace=True, errors="ignore")
df_yearly["timestamp"] = pd.to_datetime(df_yearly["timestamp"], format="%b %d, %Y")

#generate a full quarter hourly timestamp range from first year to last year
start_date = df_yearly["timestamp"].min()
end_date = df_yearly["timestamp"].max() + pd.DateOffset(years=1) #extend by 1 year to include full last year

#create a quarter hourly time range
full_time_range = pd.date_range(start=start_date, end=end_date, freq="15min")[:-1]

#create an empty df with quarter hourly timestamps with expected NaN
df_expanded_yearly = pd.DataFrame(index=full_time_range)
df_expanded_yearly.index.name = "timestamp"

#merge yearly data into quarter-hourly timestamps
df_yearly.set_index("timestamp", inplace=True)
df_expanded_yearly = df_expanded_yearly.merge(df_yearly, how="left", left_index=True, right_index=True)

#forward fill to distribute yearly values across all quarter-hourly timestamps
df_expanded_yearly.ffill(inplace=True)

df_expanded_yearly.reset_index(inplace=True)

df_expanded_yearly.to_csv("expanded_yearly.csv", index=False)

print(f"expanded yearly shape: {df_expanded_yearly.shape}")




df_quarter_hourly = pd.read_csv("merged_quarter_hourly_data.csv", parse_dates=["timestamp"], low_memory=False)
df_monthly = pd.read_csv("expanded_montly_data.csv", parse_dates=["timestamp"], low_memory=False)
df_yearly = pd.read_csv("expanded_yearly.csv", parse_dates=["timestamp"], low_memory=False)
df_hourly = pd.read_csv("expanded_hourly_data.csv", parse_dates=["timestamp"], low_memory=False)

df_quarter_hourly = df_quarter_hourly.drop_duplicates(subset=["timestamp"], keep="first")

print(df_quarter_hourly["timestamp"].duplicated().sum())
print(df_monthly["timestamp"].duplicated().sum())
print(df_yearly["timestamp"].duplicated().sum())
print(df_hourly["timestamp"].duplicated().sum())


print(f"Shape of quarter hourly: {df_quarter_hourly.shape}, shape of monthly: {df_monthly.shape}, shape of yearly: {df_yearly.shape}, shape of hourly: {df_hourly.shape}")
# Merge quarter-hourly data with expanded monthly data
merged_df = pd.merge(df_quarter_hourly, df_monthly, on="timestamp", how="inner")

# Merge the result with expanded yearly data
merged_df = pd.merge(merged_df, df_yearly, on="timestamp", how="inner")

merged_df = pd.merge(merged_df, df_hourly, on="timestamp", how="inner")

# Replace '-' with NaN
#merged_df.replace("-", np.nan, inplace=True)

# Drop columns where all values are NaN
#merged_df = merged_df.dropna(axis=1, how="all")

# Save the final merged dataset
merged_df.to_csv("final_merged_dataset.csv", index=False)


if "Germany/Luxembourg [€/MWh] Original resolutions" in merged_df.columns:
    print("found column")

print(f"final dataset merged and saved as 'final_merged_dataset.csv' with shape {merged_df.shape} and {merged_df.head()}")