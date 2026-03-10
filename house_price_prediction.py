# AI ML Task 2: House Price Prediction - Model Comparison
# This script implements an enhanced House Price Prediction system using California Housing Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 70)
print("STEP 1: Import Required Libraries")
print("=" * 70)
print("All required libraries imported successfully!")
print()

# Step 2: Load California Housing Dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['HousePrice'] = housing.target

print(data.head())

# Step 3: Separate Features (X) and Target Variable (y = "HousePrice")
X = data.drop('HousePrice', axis=1) 
y = data['HousePrice']

# Step 4: Feature Scaling using StandardScaler
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

# Step 5: Train-Test Split with test_size=0.2 and random_state=42
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train Multiple Models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

# Step 7: Model Evaluation and Comparison
print("=" * 70)
print("STEP 6 & STEP 7: Train Multiple Models & Model Evaluation")
print("=" * 70)

results = []
trained_models = {}

for name, model in models.items():
    # Train each model on training set
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics: RMSE and R-squared
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({'Model': name, 'RMSE': rmse, 'R2 Score': r2})
    print(f"{name}: RMSE = {rmse:.4f}, R2 Score = {r2:.4f}")

# Create comparison DataFrame sorted by RMSE (ascending - best first)
results_df = pd.DataFrame(results).sort_values(by='RMSE', ascending=True).reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 70)
print(results_df)

# Best performing model
best_model_name = results_df.iloc[0]['Model']
best_rmse = results_df.iloc[0]['RMSE']
best_r2 = results_df.iloc[0]['R2 Score']

print("\n" + "=" * 70)
print("BEST PERFORMING MODEL SELECTION AND JUSTIFICATION")
print("=" * 70)
print(f"Best Model Selected: {best_model_name}")
print(f"Root Mean Squared Error: {best_rmse:.4f}")
print(f"R-squared Score: {best_r2:.4f}")

# Step 8: Visual Performance Validation (Scatter Plot)
print("\n" + "=" * 70)
print("STEP 8: Visual Performance Validation")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['blue', 'green', 'orange']
markers = ['o', 's', '^']

for idx, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    ax.scatter(y_test, y_pred, c=colors[idx], marker=markers[idx], 
               alpha=0.6, s=60, label=f"{name} (R2={r2:.4f})")

# Reference line (perfect prediction)
min_val = y_test.min()
max_val = y_test.max()
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

ax.set_xlabel('Actual House Prices', fontsize=12)
ax.set_ylabel('Predicted House Prices', fontsize=12)
ax.set_title('Actual vs Predicted House Prices Comparison', fontsize=14)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nTask completed successfully!")

