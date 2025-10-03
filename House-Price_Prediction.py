#Bengaluru House Price Prediction
#Features: BHK, total_sqft, Number of bathrooms, Number of balcony

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Database

data = pd.read_csv("Bengaluru_House_Data.csv")



#Data Cleaning
# Add ID column if not present


if 'ID' not in data.columns:
    data.insert(0, 'ID', range(1, len(data) + 1))

# Extract BHK from 'size' column '2 BHK' → 2)

def extract_bhk(x):
    try:
        return int(str(x).split(' ')[0])
    except:
        return np.nan

data['BHK'] = data['size'].apply(extract_bhk)

# Clean total_sqft

def clean_sqft(x):
    try:
        if '-' in str(x):
            parts = x.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return float(x)
    except:
        return np.nan

data['total_sqft'] = data['total_sqft'].apply(clean_sqft)

# Fill missing bathroom & balcony with median values

data['bath'] = data['bath'].fillna(data['bath'].median())
data['balcony'] = data['balcony'].fillna(data['balcony'].median())

# Drop rows with missing values in key columns

data = data.dropna(subset=['BHK', 'total_sqft', 'price'])

#Select Features

features = ['BHK', 'total_sqft', 'bath', 'balcony']
X = data[features].values
y = data['price'].values  

# Target: price in Lakhs

# Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

#Train-Test Split

train_size = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:train_size], X_norm[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


#Linear Regression (Normal Equation)

X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

theta = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
y_pred = X_test_b.dot(theta)

#Model Evaluation

mse = np.mean((y_test - y_pred) ** 2)
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot Actual vs Predicted

plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

#Feature Importance

coefficients = pd.Series(theta[1:], index=features)
print("\nFeature Coefficients (Importance):")
print(coefficients.sort_values(ascending=False))


#Interactive Prediction

print("\n--- Predict Price for a New House ---")
try:
    bhk = float(input("Enter number of BHK: "))
    sqft = float(input("Enter total sqft: "))
    bath = float(input("Enter number of bathrooms: "))
    balcony = float(input("Enter number of balconies: "))

    sample = np.array([[bhk, sqft, bath, balcony]])
    sample_norm = (sample - X_mean) / X_std
    sample_b = np.c_[np.ones((1,1)), sample_norm]
    predicted_price = sample_b.dot(theta)[0]

    print(f"\n🏡 Predicted Price: ₹{predicted_price:.2f} Lakhs")

except Exception as e:
    print("❌ Error during prediction:", e)