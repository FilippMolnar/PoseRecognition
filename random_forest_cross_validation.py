import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmap plotting

# Load dataset
data = pd.read_csv("3d_keypoints.csv")

# Extract features and labels
labels = data["label"]
filenames = data["filename"]
features = data.drop(columns=["label", "filename"])

# Define number of folds
n_folds = 10  # You can change this to your desired number of folds

# Initialize StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize lists to store results
confusion_matrices = []
classification_reports = []
incorrect_predictions = []

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(features, labels)):
    print(f"Fold {fold + 1}/{n_folds}")

    # Split data into train and test sets for this fold
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    filenames_train, filenames_test = filenames.iloc[train_index], filenames.iloc[test_index]

    # Train a classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)  # added random_state for reproducibility
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # Evaluate and store results

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    classification_reports.append(report)
    print(classification_report(y_test, y_pred)) # Print report for each fold

    # Identify and store incorrect predictions
    incorrect_indices = np.where(y_pred != y_test)[0]
    incorrect_files = filenames_test.iloc[incorrect_indices]
    incorrect_labels = y_test.iloc[incorrect_indices]
    incorrect_preds = y_pred[incorrect_indices]

    incorrect_df = pd.DataFrame({
        "filename": incorrect_files.values,
        "true_label": incorrect_labels.values,
        "predicted_label": incorrect_preds
    })
    incorrect_predictions.append(incorrect_df)


# Calculate average confusion matrix
avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

# Plot the average confusion matrix
plt.figure(figsize=(10, 8))  # Adjust figure size as needed
sns.heatmap(avg_confusion_matrix, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=np.unique(labels), yticklabels=np.unique(labels))  # Use unique labels for tick labels
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Average Confusion Matrix (Cross-Validation)")
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.savefig("average_confusion_matrix.png") # Saves the plot to a file
plt.show()   # Displays the plot
print("\nAverage Confusion Matrix Plot saved to average_confusion_matrix.png")


# You can also calculate average classification metrics (precision, recall, f1-score) if needed.
# This is a bit more involved as you need to average the metrics from each fold's report.

# Concatenate incorrect predictions from all folds
all_incorrect_predictions = pd.concat(incorrect_predictions, ignore_index=True)

# Save all incorrect predictions
all_incorrect_predictions.to_csv("incorrect_predictions_cv.csv", index=False)

print("Incorrect predictions from all folds saved to incorrect_predictions_cv.csv")


# Example: Average Accuracy across folds (can extend to other metrics)
accuracies = [report['accuracy'] for report in classification_reports]
avg_accuracy = np.mean(accuracies)
print(f"\nAverage Accuracy across {n_folds} folds: {avg_accuracy:.4f}")