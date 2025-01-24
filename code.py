import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Correct the file path
file_path = r"C:\Users\barak\OneDrive\Desktop\owid-co2-data.csv"

# Step 1: Load and preprocess the dataset
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    print(f"File not found. Ensure the file exists at: {file_path}")
    exit()
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()

# Select relevant columns and drop missing values
required_columns = ['country', 'year', 'population', 'gdp', 'energy_per_capita', 'co2']
df = df[required_columns].dropna()

# Log-transform skewed features
df['population_log'] = df['population'].apply(lambda x: x if x <= 0 else np.log(x))
df['gdp_log'] = df['gdp'].apply(lambda x: x if x <= 0 else np.log(x))
df['energy_log'] = df['energy_per_capita'].apply(lambda x: x if x <= 0 else np.log(x))

# Feature-target split
X = df[['population_log', 'gdp_log', 'energy_log']]
y = df['co2']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Optimize and Train the Model
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, scoring='r2', cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

# Step 4: Predict CO2 for all countries in the dataset
df['co2_predicted'] = best_model.predict(scaler.transform(df[['population_log', 'gdp_log', 'energy_log']]))

# Add calculated metrics
df['co2_per_capita'] = df['co2'] / df['population']
df['co2_predicted_per_capita'] = df['co2_predicted'] / df['population']
df['energy_per_capita_growth'] = df.groupby('country')['energy_per_capita'].pct_change()
df['gdp_per_capita'] = df['gdp'] / df['population']

# Step 5: Load the existing predictions file
existing_predictions_file = "optimized_model_predictions.csv"
try:
    predictions_df = pd.read_csv(existing_predictions_file)
    print("Predictions file loaded successfully!")
except FileNotFoundError:
    print(f"File not found: {existing_predictions_file}")
    exit()

# Merge the enhanced metrics with the existing predictions
merged_df = pd.merge(predictions_df, df, on=['country', 'year'], how='inner')

# Step 6: Save the updated dataset
output_file = "aligned_model_prediction.csv"
merged_df.to_csv(output_file, index=False)
print(f"Enhanced dataset exported to {output_file}.")
