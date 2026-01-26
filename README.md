# Breast Cancer Classification using Logistic Regression

## Project Overview
This project implements a machine learning model to classify breast tumors as **Benign** or **Malignant** based on clinical diagnostic data. The goal was to build a highly interpretable and accurate diagnostic tool by leveraging Logistic Regression and standard data science best practices.

## Key Features
* **Full ML Pipeline**: Demonstrates data loading, exploratory analysis, feature scaling, model training, and performance evaluation.
* **Feature Standardization**: Utilizes `StandardScaler` to ensure all 30 clinical features are on the same scale, which is critical for the convergence of the Logistic Regression model.
* **Evaluation Metrics**: Uses Confusion Matrices and Classification Reports (Precision, Recall, F1-Score) to validate the model's reliability in a medical context.

## Tech Stack
* **Language**: Python
* **Libraries**: 
    * **Scikit-Learn**: For the core Logistic Regression algorithm and preprocessing.
    * **Pandas**: For data frame manipulation and analysis.
    * **Matplotlib**: For visualizing model performance and data distribution.

## Model Performance
The model achieved high accuracy on the test set, demonstrating a strong balance between sensitivity and specificity:
* **Confusion Matrix Results**:
    * Correctly identified **51** Benign cases.
    * Correctly identified **86** Malignant cases.
    * Minimal False Positives (2) and False Negatives (4).

## Dataset
The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**. It includes features such as:
* Mean Radius
* Mean Texture
* Mean Concavity
* Area and Perimeter

## Learning Resources & Credits
This project was developed as part of a learning journey in machine learning.
* **Educational Reference**: Special thanks to [DataCamp's Logistic Regression Tutorial](https://www.datacamp.com/tutorial/understanding-logistic-regression-python) for providing the foundational concepts and guidance used to build this implementation.

## How to Use
1. **Clone the repo**:
   ```bash
   git clone [https://github.com/your-username/breast-cancer-classification.git](https://github.com/your-username/breast-cancer-classification.git)
