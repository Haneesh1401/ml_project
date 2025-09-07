import pandas as pd
from sklearn.datasets import load_iris
import os

def load_data(path="data/iris.csv"):
    if os.path.exists(path):
        # Load from CSV if available
        df = pd.read_csv(path)
    else:
        # Fetch from scikit-learn if CSV not found
        iris = load_iris(as_frame=True)
        df = iris.frame
        df.rename(columns={"target": "species"}, inplace=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)  # save for future use
        print(f"Iris dataset saved to {path}")

    X = df.drop("species", axis=1)
    y = df["species"]
    return X, y
