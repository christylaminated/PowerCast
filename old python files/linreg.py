import pandas as pd

# Load the dataset
use_cols = ["Germany Price", "Netherlands (export) [MWh] Original resolutions", "Netherlands (import) [MWh] Original resolutions"]  # Update with actual column names
df = pd.read_csv("Linear_regression_data.csv", usecols=use_cols)
# Display the first few rows
#print(df.head())


# Check for missing values
#print(df.isna().sum())

# Drop rows with missing values (optional)
df.dropna(inplace=True)

# Convert categorical columns to numeric (if any exist)
df = pd.get_dummies(df)  



# Define Features (X) and Target (y)
X = df.drop(columns=["Germany Price"], axis=1)  # All columns except the target
y = df["Germany Price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize and train the model
print("about to fit")
#lr = LinearRegression()
#lr.fit(X_train, y_train)
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(max_iter=4000, tol=1e-3, verbose=1)  # Enables iteration logging
sgd.fit(X_train, y_train)

print("finished fitting")

y_pred = sgd.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

# Calculate errors and performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
