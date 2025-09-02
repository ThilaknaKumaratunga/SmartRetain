import os
import tempfile
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    base_dir = "/opt/ml/processing"
    #Read Data
    df = pd.read_csv(
        f"{base_dir}/input/churn.csv.csv"
    )
   
    # create two columns named : pay_as_you_go and contract
    df['pay_as_you_go'] = np.where(df['Tariff Plan'] == 1, 1, 0)
    df['contract'] = np.where(df['Tariff Plan'] == 2, 1, 0)
    # drop Tariff Plan
    df = df.drop('Tariff Plan', axis=1)

    # create two columns named: active and non_active
    df['active'] = np.where(df['Status'] == 1, 1, 0)
    df['non_active'] = np.where(df['Status'] == 0, 1, 0)
    df = df.drop('Status', axis=1)

    # move the Churn column to the start
    churn = df.pop('Churn')
    df.insert(0, 'Churn', churn)

    # Data Splitting
    # --- Step 1: Separate features and target ---
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    # --- Step 2: First split into train (70%) and temp (30%) ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    # --- Step 3: Split temp into validation (15%) and test (15%) ---
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )

    # --- Step 4: Combine labels and features, with label first ---
    train = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
    churn = train.pop('Churn')
    train.insert(0, 'Churn', churn)
    validation = pd.concat([y_val.reset_index(drop=True), X_val.reset_index(drop=True)], axis=1)
    churn = validation.pop('Churn')
    validation.insert(0, 'Churn', churn)
    test = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

    churn = test.pop('Churn')
    test.insert(0, 'Churn', churn)

    # --- Step 5: Save to CSV (no headers, no index) ---
    base_dir = "data"  # change this to your directory
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

    # --- Optional: check splits ---
    # print("Train size:", train.shape)
    # print("Validation size:", validation.shape)
    # print("Test size:", test.shape)
    # print("\nChurn distribution:")
    # print("Train:\n", train['Churn'].value_counts(normalize=True))
    # print("Validation:\n", validation['Churn'].value_counts(normalize=True))
    # print("Test:\n", test['Churn'].value_counts(normalize=True))


