import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# --- Configuration ---
DATA_PATH = 'creditcard.csv'
MODEL_DIR = 'artifacts'  # Folder to save our work

# Create artifacts directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_and_clean_data():
    print("Loading dataset... (This might take a few seconds)")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_PATH}. Please download it and place it in this folder.")
        return None, None

    # 1. Basic Check
    print(f"Dataset Shape: {df.shape}")
    print(f"Missing Values: {df.isnull().sum().max()}") # Should be 0

    # 2. Scaling 'Amount'
    # We scale 'Amount' because its range is very different from the PCA features (V1-V28).
    # We use StandardScaler to center it around 0 with a standard deviation of 1.
    print("Scaling 'Amount' column...")
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    # Save the scaler for later use in the Streamlit app
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("Scaler saved to artifacts/scaler.pkl")

    # 3. Dropping Unnecessary Columns
    # 'Time' is often not useful for generic fraud detection without deeper engineering, 
    # and we drop the original 'Amount' since we have 'scaled_amount'.
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # 4. Separating Features (X) and Target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    print("Data cleaning complete.")
    return X, y

if __name__ == "__main__":
    # Run the function
    X, y = load_and_clean_data()
    
    if X is not None:
        # 5. Perform the Train-Test Split here to lock it in
        print("Splitting data into Train and Test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Save the processed datasets so we don't have to clean again
        print("Saving processed data...")
        joblib.dump((X_train, X_test, y_train, y_test), os.path.join(MODEL_DIR, 'processed_data.pkl'))
        
        print("\nSUCCESS! Phase 1 complete.")
        print(f"Training Data shape: {X_train.shape}")
        print(f"Test Data shape: {X_test.shape}")
        print("You are now ready to train the model.")