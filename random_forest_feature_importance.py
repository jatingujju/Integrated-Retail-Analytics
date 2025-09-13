import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import joblib

# 1. Define the base directory
base_path = r'C:\Users\Tilak\OneDrive\Desktop\Integrated Retail Analytics for Store Optimization'

# 2. Construct the full file paths
sales_file = os.path.join(base_path, 'sales data-set.csv')
stores_file = os.path.join(base_path, 'stores data-set.csv')
features_file = os.path.join(base_path, 'Features data set.csv')

# 3. Load all three datasets
sales_df = pd.read_csv(sales_file)
stores_df = pd.read_csv(stores_file)
features_df = pd.read_csv(features_file)

# 4. Correct the date format issue using dayfirst=True
sales_df['Date'] = pd.to_datetime(sales_df['Date'], dayfirst=True)
features_df['Date'] = pd.to_datetime(features_df['Date'], dayfirst=True)

# 5. Merge the dataframes
df = pd.merge(sales_df, stores_df, on='Store', how='left')
df = pd.merge(df, features_df, on=['Store', 'Date'], how='left')

# 6. Handle the 'IsHoliday' column, as it's in two dataframes
df = df.drop(columns=['IsHoliday_y'], errors='ignore')
df.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)

# 7. Create dummy variables for 'Type'
df = pd.get_dummies(df, columns=['Type'], prefix='Type', drop_first=True)

# 8. Define the features and target
features = ['Store', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type_B', 'Type_C', 'IsHoliday']
X = df[features]
y = df['Weekly_Sales']

# 9. Split the data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train)

# 10. Save the trained model to a file
joblib.dump(rf_reg, 'random_forest_model.pkl')

# 11. Get and visualize feature importance
importances = rf_reg.feature_importances_
feature_importances_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.bar(feature_importances_df['feature'], feature_importances_df['importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance for Random Forest Model')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# 12. Print the sorted results
print("Top 10 Most Important Features:")
print(feature_importances_df.head(10))
