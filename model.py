# model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Generate dummy training data
np.random.seed(42)
data = pd.DataFrame({
    'area': np.random.randint(1000, 4000, 100),
    'bedrooms': np.random.randint(1, 5, 100),
    'age': np.random.randint(1, 30, 100)
})
data['price'] = (
    data['area'] * 300 +
    data['bedrooms'] * 50000 -
    data['age'] * 1000 +
    np.random.randint(10000, 50000, 100)
)

# Separate features and target
X = data[['area', 'bedrooms', 'age']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully as model.pkl")

