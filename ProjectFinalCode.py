# airbnb_price_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load and Clean Raw Data 
df = pd.read_csv(r"C:\Users\KIIT\Desktop\ElevateLabs\listings_clean.csv")

# Clean and engineer features
df['reviews_per_month'] = pd.to_numeric(df['reviews_per_month'], errors='coerce').fillna(0)
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['review_month'] = df['last_review'].dt.month.fillna(0).astype(int)

# Select relevant columns
model_columns = [
    'price', 'room_type', 'neighbourhood_group', 'number_of_reviews',
    'availability_365', 'minimum_nights', 'reviews_per_month', 'review_month'
]
df_model = df[model_columns]

# One-hot encoding
df_model_encoded = pd.get_dummies(df_model, columns=['room_type', 'neighbourhood_group'], drop_first=True)

# Split into features and target
X = df_model_encoded.drop('price', axis=1)
y = df_model_encoded['price']

# Save model-ready data 
X.to_csv("X_model_ready.csv", index=False)
y.to_csv("y_model_target.csv", index=False)

print(" Data is model-ready. Saved as CSV.")

# STEP 2: Model Training (Random Forest) 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_imputed, y_train)

# Predict
rf_preds = rf_model.predict(X_test_imputed)

# Save prediction results
pred_df = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': rf_preds,
    'Error': y_test.values - rf_preds
})
pred_df['Pricing Flag'] = pred_df['Error'].apply(
    lambda x: 'Overpriced' if x < -50 else 'Underpriced' if x > 50 else 'Fair'
)
pred_df.to_csv("airbnb_price_predictions.csv", index=False)
print(" Predictions saved to airbnb_price_predictions.csv")

#STEP 3: Feature Importance Plot
importances = rf_model.feature_importances_
features = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot Top 10 Predictors
plt.figure(figsize=(10,6))
sns.barplot(data=feature_importance_df.head(10), x='Importance', y='Feature')
plt.title("Top 10 Airbnb Price Predictors")
plt.tight_layout()
plt.show()

# STEP 4: Correlation Heatmap 
df_corr = X.copy()
df_corr['price'] = y
corr_matrix = df_corr.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix[['price']].sort_values(by='price', ascending=False),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    vmin=-1, vmax=1
)
plt.title("Correlation of Features with Airbnb Price")
plt.tight_layout()
plt.show()

# === Preview Predictions ===
print("\n Sample Predictions:")
print(pred_df.head(20))
