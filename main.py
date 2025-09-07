from src.data_loader import load_data
from src.evaluate import evaluate_model, cross_validate_model
from src.utils import save_model, log_results
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings

def main():
    # Ignore warnings for DummyClassifier metrics
    warnings.filterwarnings("ignore")

    # Load dataset
    X, y = load_data("data/iris.csv")

    # Define models explicitly
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogReg": LogisticRegression(max_iter=200),
        "SVM": SVC(),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        # Train model
        model.fit(X, y)
        print(f"{name} trained successfully.")

        # Evaluate model
        acc, report = evaluate_model(model, X, y, name)
        results[name] = acc

        # Cross-validation (5-fold)
        scores = cross_val_score(model, X, y, cv=5)
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV Accuracy: {scores.mean():.4f}")

        # Save trained model
        save_model(model, name)

    # Save summary
    log_results(results)
    print("\nAll models trained and results saved successfully!")


if __name__ == "__main__":
    main()
