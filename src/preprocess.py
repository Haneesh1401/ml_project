import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(X, y, test_size=0.2, random_state=42, scale=True):
    """
    Preprocess the dataset:
      - Train-test split
      - Feature scaling (optional)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    return X_train, X_test, y_train, y_test, scaler
