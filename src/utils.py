import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def save_model(model, filename):
    """Save trained model as a pickle file."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{filename}.pkl")

def load_model(filename):
    """Load model from pickle file."""
    return joblib.load(f"models/{filename}.pkl")

def save_plot(fig, filename):
    """Save matplotlib/seaborn plots."""
    os.makedirs("results", exist_ok=True)
    fig.savefig(f"results/{filename}.png", bbox_inches="tight")

def plot_confusion_matrix(cm, classes, title, filename):
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    save_plot(fig, filename)
    plt.close(fig)

def log_results(results, filename="results/summary.txt"):
    """Save accuracy results in a text file."""
    os.makedirs("results", exist_ok=True)
    with open(filename, "w") as f:
        for name, acc in results.items():
            f.write(f"{name}: {acc:.4f}\n")
