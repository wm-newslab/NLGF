import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import krippendorff

# ----------------------------------------------------
# 1. Load Data
# ----------------------------------------------------

df = pd.read_csv("data_labels.csv")

# Your CSV must contain:
# id, rater1_label, rater2_label

print(df.head())


# ----------------------------------------------------
# 2. Extract label columns
# ----------------------------------------------------

r1 = df["gaga-gfl"]
r2 = df["kate-gfl"]


# ----------------------------------------------------
# 3. Percentage Agreement
# ----------------------------------------------------

agreement = (r1 == r2).mean()
print("Percentage Agreement:", agreement)


# ----------------------------------------------------
# 4. Cohen's Kappa
# ----------------------------------------------------

kappa = cohen_kappa_score(r1, r2)
print("Cohen's Kappa:", kappa)


# ----------------------------------------------------
# 5. Krippendorff's Alpha (Nominal)
# ----------------------------------------------------

data_matrix = [
    r1.tolist(),
    r2.tolist()
]

alpha = krippendorff.alpha(reliability_data=data_matrix,
                           level_of_measurement="nominal")

print("Krippendorff's Alpha:", alpha)


# ----------------------------------------------------
# 6. Confusion Matrix
# ----------------------------------------------------

labels = sorted(set(r1) | set(r2))   # union of all labels
cm = confusion_matrix(r1, r2, labels=labels)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\nConfusion Matrix:\n", cm_df)


# ----------------------------------------------------
# 7. Plot Confusion Matrix
# ----------------------------------------------------

plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: Rater 1 vs Rater 2")
plt.xlabel("Rater 2")
plt.ylabel("Rater 1")
plt.tight_layout()
plt.show()


# ----------------------------------------------------
# 8. Label-wise Agreement Statistics
# ----------------------------------------------------

label_stats = {}

for label in labels:
    total = ((r1 == label) | (r2 == label)).sum()
    agree = ((r1 == label) & (r2 == label)).sum()
    rate = agree / total if total > 0 else 0
    label_stats[label] = {
        "times_labeled_by_either": total,
        "times_both_agreed": agree,
        "agreement_rate": rate
    }

label_stats_df = pd.DataFrame(label_stats).T
print("\nLabel-wise Agreement Statistics:\n", label_stats_df)


# ----------------------------------------------------
# 9. Visualization of Disagreements
# ----------------------------------------------------

df["disagree"] = (df["gaga-gfl"] != df["kate-gfl"])
disagreement_counts = df[df["disagree"]]["gaga-gfl"].value_counts()

plt.figure(figsize=(8,6))
disagreement_counts.plot(kind="bar")
plt.title("Which Labels Rater 1 Used When They Disagreed")
plt.ylabel("Count")
plt.xlabel("Label")
plt.tight_layout()
plt.show()
