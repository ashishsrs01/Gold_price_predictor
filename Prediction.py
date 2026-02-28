import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load the CSV file
df = pd.read_csv('gold_prices_1995-2026.csv')

# Display basic info
print("Dataset Overview:")
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
df = df.dropna()

# Convert date column to datetime if exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Remove duplicates
df = df.drop_duplicates()

# Feature engineering: Create new features if necessary
if 'Open' in df.columns and 'Close' in df.columns:
    df['Price Change'] = df['Close'] - df['Open']

# ---------------------------------------------------------
# Advanced Feature Engineering for Better Accuracy
# ---------------------------------------------------------

# Extract date features to capture temporal patterns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['DayOfYear'] = df['Date'].dt.dayofyear
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Create lag features (previous prices)
df['Price_Lag1'] = df['Gold_Price_USD_YFinance'].shift(1)
df['Price_Lag3'] = df['Gold_Price_USD_YFinance'].shift(3)
df['Price_Lag6'] = df['Gold_Price_USD_YFinance'].shift(6)
df['Price_Lag12'] = df['Gold_Price_USD_YFinance'].shift(12)

# Create moving averages
df['MA3'] = df['Gold_Price_USD_YFinance'].rolling(window=3).mean()
df['MA6'] = df['Gold_Price_USD_YFinance'].rolling(window=6).mean()
df['MA12'] = df['Gold_Price_USD_YFinance'].rolling(window=12).mean()

# Create volatility feature
df['Volatility'] = df['Gold_Price_USD_YFinance'].rolling(window=6).std()

# Drop NaN rows created by lag and rolling features
df = df.dropna()

# ---------------------------------------------------------
# Preprocessing and split data for training
# ---------------------------------------------------------

# Separate features and target
X = df.drop(['Gold_Price_USD_YFinance', 'Date'], axis=1)
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

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Number of features: {X_train.shape[1]}")

# ---------------------------------------------------------
# Train and evaluate multiple models
# ---------------------------------------------------------

print("\n" + "="*60)
print("TRAINING AND COMPARING MULTIPLE MODELS")
print("="*60)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR (RBF)': SVR(kernel='rbf', C=100, gamma='scale')
}

results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    results[model_name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train
    }
    
    print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | Test MAE: ${test_mae:.2f}")

# ---------------------------------------------------------
# Select best model
# ---------------------------------------------------------

best_model_name = max(results, key=lambda x: results[x]['test_r2'])
best_model_data = results[best_model_name]
best_model = best_model_data['model']

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print("="*60)

y_pred_test = best_model_data['y_pred_test']
y_pred_train = best_model_data['y_pred_train']

print("\nPredictions on test set (first 10 samples):")
print(pd.DataFrame({
    'Actual': y_test.head(10).values,
    'Predicted': y_pred_test[:10]
}))

# ---------------------------------------------------------
# Evaluate best model performance
# ---------------------------------------------------------

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Training metrics
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = best_model_data['train_r2']

# Test metrics
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = best_model_data['test_rmse']
test_mae = best_model_data['test_mae']
test_r2 = best_model_data['test_r2']

print("\nTraining Metrics:")
print(f"  MSE (Mean Squared Error): {train_mse:.4f}")
print(f"  RMSE (Root Mean Squared Error): {train_rmse:.4f}")
print(f"  MAE (Mean Absolute Error): {train_mae:.4f}")
print(f"  R² Score: {train_r2:.4f}")

print("\nTest Metrics:")
print(f"  MSE (Mean Squared Error): {test_mse:.4f}")
print(f"  RMSE (Root Mean Squared Error): {test_rmse:.4f}")
print(f"  MAE (Mean Absolute Error): {test_mae:.4f}")
print(f"  R² Score: {test_r2:.4f}")

# ---------------------------------------------------------
# Model Comparison Summary
# ---------------------------------------------------------

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train R²': [results[m]['train_r2'] for m in results.keys()],
    'Test R²': [results[m]['test_r2'] for m in results.keys()],
    'Test MAE': [results[m]['test_mae'] for m in results.keys()],
    'Test RMSE': [results[m]['test_rmse'] for m in results.keys()]
})

print("\n", comparison_df.to_string(index=False))

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total samples processed: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features used: {X_train.shape[1]}")
print(f"\nBest Model: {best_model_name}")
print(f"Best Model Performance (Test Set):")
print(f"  Accuracy (R² Score): {test_r2:.2%}")
print(f"  Average Error: ${test_mae:.2f}")
print(f"  Root Mean Squared Error: ${test_rmse:.2f}")

# ---------------------------------------------------------
# Future Price Prediction (Next 12 Months) - Recursive Approach
# ---------------------------------------------------------

print("\n" + "="*60)
print("PREDICTING FUTURE GOLD PRICES (Next 12 Months)")
print("="*60)

# Get the last date and price in the dataset
last_date = df['Date'].max()
last_price = df['Gold_Price_USD_YFinance'].iloc[-1]
print(f"\nLast known date: {last_date.strftime('%Y-%m-%d')}")
print(f"Last known price: ${last_price:,.2f}\n")

# Get recent historical prices for lag features
recent_prices = df['Gold_Price_USD_YFinance'].tail(12).values

# Create future dates (next 12 months)
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')

# Initialize prediction list and price history
future_predictions = []
price_history = list(recent_prices) + [last_price]  # Keep historical context

for future_date in future_dates:
    # Create feature row for future date
    feature_row = pd.DataFrame({
        'Year': [future_date.year],
        'Month': [future_date.month],
        'Quarter': [future_date.quarter],
        'DayOfYear': [future_date.dayofyear],
        'WeekOfYear': [future_date.isocalendar()[1]]
    })
    
    # Use actual recent prices for lag features (recursive approach)
    feature_row['Price_Lag1'] = [price_history[-1]]       # Last predicted or actual price
    feature_row['Price_Lag3'] = [price_history[-3]]       # 3 months ago
    feature_row['Price_Lag6'] = [price_history[-6]]       # 6 months ago
    feature_row['Price_Lag12'] = [price_history[-12]]     # 12 months ago
    
    # Calculate moving averages from recent prices
    feature_row['MA3'] = [np.mean(price_history[-3:])]    # 3-month MA
    feature_row['MA6'] = [np.mean(price_history[-6:])]    # 6-month MA
    feature_row['MA12'] = [np.mean(price_history[-12:])]  # 12-month MA
    
    # Volatility from recent prices
    feature_row['Volatility'] = [np.std(price_history[-6:])]
    
    # Scale the features
    feature_row_scaled = scaler.transform(feature_row)
    feature_row_scaled = pd.DataFrame(feature_row_scaled, columns=X.columns)
    
    # Make prediction
    predicted_price = best_model.predict(feature_row_scaled)[0]
    
    # Store prediction for next iteration (feedback loop)
    price_history.append(predicted_price)
    
    future_predictions.append({
        'Date': future_date,
        'Predicted_Price': predicted_price
    })

# Create DataFrame for future predictions
future_df = pd.DataFrame(future_predictions)

print("Future Gold Price Predictions:")
print("-" * 60)
for idx, row in future_df.iterrows():
    print(f"{row['Date'].strftime('%B %Y'):20} -> ${row['Predicted_Price']:,.2f}")

# ---------------------------------------------------------
# Average predicted price and trend
# ---------------------------------------------------------

print("\n" + "-" * 60)
avg_future_price = future_df['Predicted_Price'].mean()
current_price = df['Gold_Price_USD_YFinance'].iloc[-1]
price_change = avg_future_price - current_price
price_change_pct = (price_change / current_price) * 100

print(f"Current Price (Latest): ${current_price:,.2f}")
print(f"Average Predicted Price (Next 12 Months): ${avg_future_price:,.2f}")
print(f"Predicted Change: ${price_change:,.2f} ({price_change_pct:+.2f}%)")

if price_change > 0:
    print(f"\n📈 Gold prices are expected to INCREASE over the next 12 months")
elif price_change < -5:
    print(f"\n📉 Gold prices are expected to DECREASE over the next 12 months")
else:
    print(f"\n➡️ Gold prices are expected to STABILIZE over the next 12 months")

print("\n" + "="*60)
