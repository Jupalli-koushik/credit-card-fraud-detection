import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- Configuration ---
MODEL_DIR = 'artifacts'
PROCESSED_DATA_PATH = os.path.join(MODEL_DIR, 'processed_data.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'fraud_model.pkl')

def train():
    # 1. Load Data
    print("Loading processed data...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: {PROCESSED_DATA_PATH} not found. Run data_prep.py first.")
        return

    X_train, X_test, y_train, y_test = joblib.load(PROCESSED_DATA_PATH)
    
    # 2. Handle Imbalance (SMOTE)
    # This creates synthetic fraud cases so the model doesn't just memorize "Not Fraud"
    print("Applying SMOTE to balance the training set... (This might take 1-2 minutes)")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print(f"Original Fraud Counts: {sum(y_train)}")
    print(f"New Balanced Fraud Counts: {sum(y_train_res)}")

    # 3. Train the Model
    print("Training Random Forest Model... (Go grab a coffee, this takes 2-5 mins)")
    # n_jobs=-1 uses all your CPU cores to speed it up
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    # 4. Evaluate
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Save the Model
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("SUCCESS! Model saved. Day 1 is complete.")

if __name__ == "__main__":
    train()