import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("3d_keypoints.csv")

# Extract features and labels
labels = data["label"]
filenames = data["filename"]
features = data.drop(columns=["label", "filename"])

# Split data into train and test sets
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    features, labels, filenames, test_size=0.2, random_state=42, stratify=labels
)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

import pickle
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Predictions
with open("random_forest_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test)

# Evaluate accuracy per class
report = classification_report(y_test, y_pred, output_dict=True)
print("Accuracy per class:")
for label, metrics in report.items():
    if isinstance(metrics, dict):
        print(f"{label}: {metrics['precision']:.2f}")

# Identify incorrect predictions
incorrect_indices = np.where(y_pred != y_test)[0]
incorrect_files = filenames_test.iloc[incorrect_indices]
incorrect_labels = y_test.iloc[incorrect_indices]
incorrect_preds = y_pred[incorrect_indices]

# Save incorrect predictions
incorrect_df = pd.DataFrame({
    "filename": incorrect_files.values,
    "true_label": incorrect_labels.values,
    "predicted_label": incorrect_preds
})
incorrect_df.to_csv("incorrect_predictions.csv", index=False)

print("Incorrect predictions saved to incorrect_predictions.csv")