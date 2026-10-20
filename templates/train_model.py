import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Dummy dataset for model training
np.random.seed(42)
n_samples = 500

# Features: Age, Glucose, BloodPressure, Cholesterol
X = np.random.randint(20, 80, size=(n_samples, 4))
X[:, 1] = np.random.randint(70, 200, size=n_samples) # Glucose
X[:, 2] = np.random.randint(60, 140, size=n_samples) # BP
X[:, 3] = np.random.randint(150, 300, size=n_samples) # Cholesterol

# Medical logic rules
y_heart = ((X[:, 0] > 50) & (X[:, 2] > 90) | (X[:, 3] > 240)).astype(int)
y_diabetes = ((X[:, 1] > 140) | (X[:, 0] > 45)).astype(int)

# Train Real Classifiers
model_heart = RandomForestClassifier(n_estimators=10).fit(X, y_heart)
model_diabetes = RandomForestClassifier(n_estimators=10).fit(X, y_diabetes)

# Save Models to 'models' folder
os.makedirs('models', exist_ok=True)
with open('models/heart_model.pkl', 'wb') as f:
    pickle.dump(model_heart, f)
with open('models/diabetes_model.pkl', 'wb') as f:
    pickle.dump(model_diabetes, f)
