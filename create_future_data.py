import pandas as pd
import os
import joblib

# Define the base directory and file paths
base_path = r'C:\Users\Tilak\OneDrive\Desktop\Integrated Retail Analytics for Store Optimization'
future_data_file = os.path.join(base_path, 'future_features.csv')
stores_file = os.path.join(base_path, 'stores data-set.csv')

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Load the future features and stores data
future_features_df = pd.read_csv(future_data_file)
stores_df = pd.read_csv(stores_file)

# Prepare the data for prediction (same steps as training)
# Correct the date format
future_features_df['Date'] = pd.to_datetime(future_features_df['Date'], dayfirst=True)

# Merge with stores data
future_features_df = pd.merge(future_features_df, stores_df, on='Store', how='left')

# Create dummy variables for 'Type'
future_features_df = pd.get_dummies(future_features_df, columns=['Type'], prefix='Type', drop_first=True)

# Align columns to match the training data
features = ['Store', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type_B', 'Type_C', 'IsHoliday']
X_predict = future_features_df[features]

# Make the predictions
predictions = model.predict(X_predict)

# Add the predictions to your dataframe
future_features_df['Predicted_Weekly_Sales'] = predictions

# Display the results
print(future_features_df[['Store', 'Date', 'Predicted_Weekly_Sales']])

# You can also save the results to a CSV file
output_file = os.path.join(base_path, 'forecasted_sales.csv')
future_features_df.to_csv(output_file, index=False)
print(f'\nForecasted sales saved to: {output_file}')