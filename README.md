

# bug_prediction.py

"""
A simple AI-powered script to predict whether a software commit may introduce a bug.
Demonstrates automation, decision-making enhancement, and problem-solving in software development.

Data columns (simulated):
- lines_added
- lines_removed
- files_changed
- commit_message_length
- bug (0 = no bug, 1 = likely bug)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Generate a mock dataset
np.random.seed(42)
data = pd.DataFrame({
    'lines_added': np.random.poisson(10, 1000),
    'lines_removed': np.random.poisson(5, 1000),
    'files_changed': np.random.randint(1, 10, 1000),
    'commit_message_length': np.random.randint(10, 200, 1000),
    'bug': np.random.binomial(1, 0.3, 1000)  # Simulated bug labels
})

# 2. Preprocessing
X = data.drop('bug', axis=1)
y = data['bug']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Predict & evaluate
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Bug Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. Feature importance
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
feature_importances.sort_values().plot(kind='barh', title='Feature Importance')
plt.show()
