

Readme · MD
# Breast Cancer Classification using Logistic Regression
 
## Project Overview
This project classifies breast tumours as **Benign** or **Malignant** based on clinical diagnostic data. It includes both a scikit-learn baseline and a **from-scratch implementation of Logistic Regression** built in NumPy, developed to demonstrate a first-principles understanding of how the model learns. The goal was to build an interpretable and accurate diagnostic tool while implementing the underlying algorithm directly rather than relying solely on a library.
 
## Key Features
 
* **From-Scratch Logistic Regression:** A custom implementation of Logistic Regression in NumPy, covering the sigmoid activation, binary cross-entropy cost function, and batch gradient descent. The core algorithm uses no machine learning libraries.
* **Validated Against Scikit-Learn:** The custom model is benchmarked against `sklearn`'s `LogisticRegression` on an identical train/test split to verify correctness.
* **Full ML Pipeline:** Demonstrates data loading, feature scaling, model training, and performance evaluation.
* **Feature Standardization:** Utilises `StandardScaler`, fit on the training set only and applied to the test set, ensuring all 30 clinical features are on the same scale (critical for gradient descent convergence) while avoiding data leakage.
* **Evaluation Metrics:** Uses Confusion Matrices and Classification Reports (Precision, Recall, F1-Score) to validate the model's reliability in a medical context, with particular attention to false negatives.
## From-Scratch Implementation
The custom model implements Logistic Regression from first principles:
 
* **Forward Pass:** `p = sigmoid(X · w + b)`
* **Cost Function:** Binary cross-entropy, `-mean(y·log(p) + (1-y)·log(1-p))`
* **Gradients:** Derived via the chain rule, simplifying to `dw = Xᵀ(p - y) / n` and `db = mean(p - y)`
* **Optimisation:** Batch gradient descent with a configurable learning rate and iteration count
The class mirrors the scikit-learn interface (`fit`, `predict`, `predict_proba`) so it can be evaluated with the same tooling as the baseline.
 
## Tech Stack
 
* **Language:** Python
* **Libraries:**
    * **NumPy:** For the from-scratch Logistic Regression implementation.
    * **Scikit-Learn:** For the baseline model, dataset, preprocessing, and evaluation metrics.
    * **Pandas:** For data frame manipulation and analysis.
    * **Matplotlib & Seaborn:** For visualising model performance.
## Model Performance
 
### Scikit-Learn Baseline
The baseline model achieved high accuracy on the test set, with a strong balance between sensitivity and specificity:
 
```
              precision    recall  f1-score   support
 
      Benign       0.93      0.94      0.93        53
      Malign       0.97      0.96      0.96        90
 
    accuracy                           0.95       143
   macro avg       0.95      0.95      0.95       143
weighted avg       0.95      0.95      0.95       143
```

The from-scratch implementation achieves performance in line with the scikit-learn baseline, confirming that the gradient descent training is correct. Minor differences are expected, as scikit-learn applies L2 regularisation by default whereas the custom model uses unregularised gradient descent.
 
## Dataset
The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, containing 30 numeric features per sample, including:
 
* Mean Radius
* Mean Texture
* Mean Concavity
* Area and Perimeter
## Learning Resources & Credits
The initial scikit-learn baseline was developed with reference to [DataCamp's Logistic Regression Tutorial](https://www.datacamp.com/tutorial/understanding-logistic-regression-python). The from-scratch implementation was built independently from the underlying mathematics.
 
## How to Use
 
1. Clone the repository:
```
    git clone https://github.com/roodruh/breast-cancer-classification.git
    cd breast-cancer-classification
```
 
2. Install dependencies:
```
    pip install -r requirements.txt
```
 
3. Run the classifier:
```
    python main.py
```
 
