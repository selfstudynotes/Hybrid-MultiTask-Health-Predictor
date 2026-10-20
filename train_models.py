import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("⏳ Training Heart Disease Model...")
try:
    heart_df = pd.read_csv('heart.csv')
    
    # Auto-detect target column for Heart dataset
    if 'target' in heart_df.columns:
        y_heart = heart_df['target']
    elif 'num' in heart_df.columns:
        y_heart = heart_df['num'].apply(lambda x: 1 if x > 0 else 0)
    else:
        y_heart = heart_df.iloc[:, -1]

    # Select features (Age, Resting BP, Cholesterol)
    if 'trestbps' in heart_df.columns and 'chol' in heart_df.columns:
        X_heart = heart_df[['age', 'trestbps', 'chol']]
    else:
        X_heart = heart_df.iloc[:, [0, 3, 4]]

    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

    heart_model = RandomForestClassifier(n_estimators=100, random_state=42)
    heart_model.fit(X_train_h, y_train_h)

    heart_acc = accuracy_score(y_test_h, heart_model.predict(X_test_h))
    print(f"✅ Heart Model Trained! Accuracy: {heart_acc * 100:.2f}%")

    with open('heart_model.pkl', 'wb') as f:
        pickle.dump(heart_model, f)

except Exception as e:
    print(f"❌ Error in Heart Model Training: {e}")

print("\n⏳ Training Diabetes Model...")
try:
    diabetes_df = pd.read_csv('diabetes.csv')
    X_diabetes = diabetes_df[['Age', 'Glucose', 'BloodPressure']]
    y_diabetes = diabetes_df['Outcome']

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

    diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
    diabetes_model.fit(X_train_d, y_train_d)

    diabetes_acc = accuracy_score(y_test_d, diabetes_model.predict(X_test_d))
    print(f"✅ Diabetes Model Trained! Accuracy: {diabetes_acc * 100:.2f}%")

    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(diabetes_model, f)

    print("\n🎉 BOTH REAL MACHINE LEARNING MODELS TRAINED SUCCESSFULLY!")

except Exception as e:
    print(f"❌ Error in Diabetes Model Training: {e}")