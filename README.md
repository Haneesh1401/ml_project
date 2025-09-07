# ML Project: Iris Flower Classification

## Project Overview
This project demonstrates a **Machine Learning workflow** using the **Iris dataset**. The goal is to classify iris flowers into three species based on their **sepal and petal measurements**.  

The project includes:
- Data loading and preprocessing  
- Training multiple ML models: DummyClassifier, Logistic Regression, SVM, Random Forest  
- Evaluating models with accuracy, classification report, and cross-validation  
- Visualizing **feature importance** for Random Forest  
- Saving models and results for future use  

## Dataset

- **Iris Dataset**: 150 samples, 4 features, 3 classes  
- Features:

| Feature         | Description               |
|-----------------|--------------------------|
| sepal_length    | Sepal length in cm       |
| sepal_width     | Sepal width in cm        |
| petal_length    | Petal length in cm       |
| petal_width     | Petal width in cm        |

- Target:

| Class | Species              |
|-------|---------------------|
| 0     | Iris-setosa          |
| 1     | Iris-versicolor      |
| 2     | Iris-virginica       |




## Exploratory Data Analysis

We provide a Jupyter Notebook `exploratory.ipynb` to explore the Iris dataset before training models.  

This notebook includes:
- Viewing the first few rows of the dataset
- Checking dataset info and statistical summary
- Visualizing **class distribution**
- Plotting **pairwise feature relationships**
- Showing **feature correlation heatmap**
- Saving a cleaned version of the dataset (optional)

To run the notebook:

```bash
jupyter notebook exploratory.ipynb





## Installation

1. Clone the repository:

```bash
git clone https://github.com/Haneesh1401/ml_project.git
cd ml_project
Create a virtual environment:

bash
Copy code
python -m venv venv
Activate the virtual environment:

Windows (PowerShell):

powershell
Copy code
.\venv\Scripts\activate
Linux / Mac:

bash
Copy code
source venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt



## How to Run

```bash
python -m main
The script will:

Load the Iris dataset

Train the following models:

DummyClassifier (baseline)

Logistic Regression

SVM

Random Forest

Print accuracy and classification reports

Perform 5-fold cross-validation

Save trained models in models/

Save results summary in results/summary.txt


## Sample Output

Dummy trained successfully.
Model: Dummy, Accuracy: 0.3000
Cross-validation scores: [0.33 0.33 0.33 0.33 0.33]
Mean CV Accuracy: 0.3333

LogReg trained successfully.
Model: LogReg, Accuracy: 1.0000
Cross-validation scores: [0.9667 1.0000 0.9333 0.9667 1.0000]
Mean CV Accuracy: 0.9733

SVM trained successfully.
RandomForest trained successfully.
Model: RandomForest, Accuracy: 1.0000
Feature importance plotted for RandomForest.


## Key Features

- **Multiple ML algorithms** to compare performance  
- **Cross-validation** to assess model robustness  
- **Feature importance visualization** (Random Forest)  
- **Save/load trained models** for reuse  
- Easy to extend for other datasets or models  