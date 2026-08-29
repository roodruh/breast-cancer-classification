import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from regression import LogisticRegression

data = load_breast_cancer(as_frame=True)
X = data['data']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(lr=0.1, n_iters=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)

cnf_matrix = confusion_matrix(y_test, preds)

class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(2)
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

cnf_df = pd.DataFrame(cnf_matrix)
sns.heatmap(cnf_df, annot=True, cmap="YlGnBu", fmt="g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion Matrix")
plt.ylabel("Actual Result")
plt.xlabel("Predicted Result")
plt.show()

target_names = ["Benign", "Malign"]
print(classification_report(y_test, preds, target_names=target_names))