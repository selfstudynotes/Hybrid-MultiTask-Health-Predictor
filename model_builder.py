import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Attention, Concatenate
from sklearn.preprocessing import StandardScaler

print("⏳ Training Real Hybrid Multi-Task Deep Neural Network...")

# 1. Dataset Generation (Age, Glucose, Sys_BP, Dia_BP, Cholesterol)
np.random.seed(42)
X_data = np.random.uniform(low=[20, 70, 90, 60, 150], high=[80, 250, 180, 110, 300], size=(1000, 5))

# Clinical Rules (Heart & Diabetes Risk Targets)
y_heart = ((X_data[:, 2] >= 130) | (X_data[:, 3] >= 80) | (X_data[:, 4] >= 200)).astype(int)
y_diabetes = (X_data[:, 1] >= 126).astype(int)

# 2. Scikit-Learn Feature Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# Save Scaler Object
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 3. Keras Deep Learning Multi-Task Model with Attention Mechanism
inputs = Input(shape=(5,), name="clinical_features")

# Shared Dense Representation Layer
shared_dense = Dense(32, activation='relu')(inputs)

# Self-Attention Layer Implementation
reshaped = tf.keras.layers.Reshape((1, 32))(shared_dense)
attn_out = Attention()([reshaped, reshaped])
attn_flat = tf.keras.layers.Flatten()(attn_out)

# Joint Feature Fusion
fused = Concatenate()([shared_dense, attn_flat])

# Task Branch 1: Cardiovascular Risk Branch
heart_dense = Dense(16, activation='relu')(fused)
heart_output = Dense(1, activation='sigmoid', name='heart_output')(heart_dense)

# Task Branch 2: Diabetes Risk Branch
diabetes_dense = Dense(16, activation='relu')(fused)
diabetes_output = Dense(1, activation='sigmoid', name='diabetes_output')(diabetes_dense)

# Multi-Task Joint Model Construction
multitask_model = Model(inputs=inputs, outputs=[heart_output, diabetes_output])

# FIXED: Provided accuracy metric for both outputs
multitask_model.compile(
    optimizer='adam',
    loss={'heart_output': 'binary_crossentropy', 'diabetes_output': 'binary_crossentropy'},
    metrics={'heart_output': 'accuracy', 'diabetes_output': 'accuracy'}
)

# Train Model
multitask_model.fit(X_scaled, {'heart_output': y_heart, 'diabetes_output': y_diabetes}, epochs=10, batch_size=16, verbose=0)

# Save Trained Model File
multitask_model.save('multitask_model.h5')
print("✅ Multi-Task Neural Network & Scaler Saved Successfully!")