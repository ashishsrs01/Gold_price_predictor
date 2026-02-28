import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv('gold_prices_1995-2026.csv')

# Display basic info
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
df = df.dropna()  # or use df.fillna(method='ffill') for forward fill

# Convert date column to datetime if exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Remove duplicates
df = df.drop_duplicates()

# Feature engineering: Create new features if necessary
# For example, if we have 'Open' and 'Close' prices, we can create a 'Price Change' feature
if 'Open' in df.columns and 'Close' in df.columns:
    df['Price Change'] = df['Close'] - df['Open']

# ---------------------------------------------------------
# continue preprocessing and split data for training
# ---------------------------------------------------------

# Separate features and target
X = df.drop('Gold_Price_USD_YFinance', axis=1)
y = df['Gold_Price_USD_YFinance']

# Encode categorical variables if any
X = pd.get_dummies(X, drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# At this point you can train your model using X_train and y_train
# Example training code:
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X_train, y_train)
   

