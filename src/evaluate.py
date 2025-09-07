import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
import os

def evaluate_model(model, X_test, y_test, model_name, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Classification Report
    report = classification_report(y_test, y_pred)
    print(f"Model: {model_name}, Accuracy: {acc:.4f}")
    print(report)

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    return acc, report

def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f}")
    return scores
