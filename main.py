
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from cleanlab.filter import find_label_issues
from cleanlab.outlier import OutOfDistribution

# 1. Generate a synthetic dataset
print("Generating synthetic dataset...")
X, y_true = make_classification(
    n_samples=500, 
    n_features=5, 
    n_informative=3, 
    n_redundant=1, 
    n_classes=3, 
    random_state=42
)

# 2. Introduce label noise (mislabeling)
y_noisy = y_true.copy()
# Flip 15% of the labels randomly
np.random.seed(42)
noise_indices = np.random.choice(len(y_true), size=int(0.15 * len(y_true)), replace=False)
for idx in noise_indices:
    # Change label to a random other class
    y_noisy[idx] = np.random.choice([l for l in range(3) if l != y_true[idx]])

print(f"Original Dataset Accuracy (with noise): {accuracy_score(y_true, y_noisy):.2f}")
print(f"Number of mislabeled examples introduced: {len(noise_indices)}")

# 3. Train a classifier and get predicted probabilities (out-of-sample)
print("\nTraining classifier and getting probabilities...")
clf = LogisticRegression(max_iter=1000, random_state=42)
# We use cross_val_predict to get clean probabilities for the training set itself
pred_probs = cross_val_predict(clf, X, y_noisy, cv=5, method="predict_proba")

# 4. Find label issues using Cleanlab
print("\nRunning Cleanlab to find label issues...")
label_issues_indices = find_label_issues(
    labels=y_noisy,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence"
)

print(f"Cleanlab found {len(label_issues_indices)} potential label issues.")
print(f"Top 5 indices flagged by Cleanlab: {label_issues_indices[:5]}")

# Calculate how many of the found issues were actually artificially flipped
correctly_found = len(set(label_issues_indices).intersection(set(noise_indices)))
precision = correctly_found / len(label_issues_indices) if len(label_issues_indices) > 0 else 0.0
recall = correctly_found / len(noise_indices) if len(noise_indices) > 0 else 0.0

print(f"Precision of Cleanlab (on artificial noise): {precision:.2f}")
print(f"Recall of Cleanlab (on artificial noise): {recall:.2f}")

# 5. Outlier Detection (Out-of-Distribution)
# Cleanlab can also detect OOD examples if we consider 'outliers' as samples with low confidence or strange features.
# A simpler way using cleanlab's OutOfDistribution class specifically for OOD:
print("\nRunning Outlier/OOD detection...")

# We calculate feature embeddings or use raw features if they are meaningful.
# For this simple example, we use the raw features X.
# Note: In a real image scenario, you would use embeddings from a pre-trained net (like ResNet/ViT).

ood = OutOfDistribution()
# We can use the already calculated probabilities or features.
# 'train_deviations' gives a score: lower score = more likely to be an outlier
ood_scores = ood.fit_score(features=X, labels=y_noisy)

# Let's say we want to check the bottom 1% scores
threshold = np.percentile(ood_scores, 1) # 1st percentile
potential_outliers = np.where(ood_scores < threshold)[0]

print(f"Potential OOD/Outliers found (bottom 1% scores): {len(potential_outliers)}")
print(f"Indices of potential outliers: {potential_outliers}")

print("\nCleaning process simulation complete.")
