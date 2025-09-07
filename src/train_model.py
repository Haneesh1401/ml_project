from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

def get_models():
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogReg": LogisticRegression(max_iter=200),
        "SVM": SVC(kernel="rbf", probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    return models

def plot_feature_importance(model, feature_names, model_name, results_dir="results"):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    if hasattr(model, "feature_importances_"):
        os.makedirs(results_dir, exist_ok=True)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(6, 4))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.title(f"Feature Importances - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{model_name}_feature_importance.png"))
        plt.close()
