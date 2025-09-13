import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- Data Loading and Merging ---
try:
    sales_df = pd.read_csv('sales data-set.csv')
    features_df = pd.read_csv('Features data set.csv')
    stores_df = pd.read_csv('stores data-set.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure all required CSV files are in the same directory as your script.")
    exit()

# Convert 'Date' column to datetime
sales_df['Date'] = pd.to_datetime(sales_df['Date'], format='%d/%m/%Y')
features_df['Date'] = pd.to_datetime(features_df['Date'], format='%d/%m/%Y')

# Merge dataframes
merged_df = pd.merge(sales_df, features_df, on=['Store', 'Date'], how='left')
merged_df = pd.merge(merged_df, stores_df, on='Store', how='left')

# Drop the redundant 'IsHoliday_y' column
merged_df.drop('IsHoliday_y', axis=1, inplace=True)
merged_df.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)

# --- Handling Missing and Invalid Values ---
merged_df.dropna(subset=['Weekly_Sales'], inplace=True)
merged_df = merged_df.loc[merged_df['Weekly_Sales'] > 0]

# Fill missing values
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
for col in markdown_cols:
    merged_df[col] = merged_df[col].fillna(0)
merged_df['CPI'] = merged_df['CPI'].fillna(merged_df['CPI'].mean())
merged_df['Unemployment'] = merged_df['Unemployment'].fillna(merged_df['Unemployment'].mean())

# --- Feature Engineering and Preprocessing ---
merged_df['Weekly_Sales_Log'] = np.log1p(merged_df['Weekly_Sales'])
merged_df['Year'] = merged_df['Date'].dt.year
merged_df['Month'] = merged_df['Date'].dt.month
merged_df['Week'] = merged_df['Date'].dt.isocalendar().week.astype(int)

# Select features and target variable
features = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 
            'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 
            'CPI', 'Unemployment', 'Size', 'Year', 'Month', 'Week', 'Type']
target = 'Weekly_Sales_Log'
X = merged_df[features]
y = merged_df[target]
X = pd.get_dummies(X, columns=['Type'], drop_first=True)
X['IsHoliday'] = X['IsHoliday'].astype(int)

# --- Model Training and Prediction ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_log = rf_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

# --- Visualization ---
plt.figure(figsize=(10, 8))
plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted Weekly Sales (Random Forest)', fontsize=16)
plt.xlabel('Actual Weekly Sales ($)', fontsize=12)
plt.ylabel('Predicted Weekly Sales ($)', fontsize=12)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2, label='Perfect Fit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('actual_vs_predicted_sales_rf.png')
print("\nScatter plot of actual vs. predicted sales saved as 'actual_vs_predicted_sales_rf.png'")
