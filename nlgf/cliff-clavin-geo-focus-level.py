import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def parse_list(v):
    if isinstance(v, list):
        return v
    if pd.isna(v):
        return []
    if isinstance(v, str):
        v = v.strip()
        if v in ("", "[]"):
            return []
        try:
            return ast.literal_eval(v)
        except:
            return []
    return []

def is_empty_list(v):
    return isinstance(v, list) and len(v) == 0

def classify(row):
    cities = parse_list(row['geo_cities'])
    states = parse_list(row['geo_states'])
    countries = parse_list(row['geo_countries'])
    pub_state = row['publisher_state_geoid']

    no_city = is_empty_list(cities)
    no_state = is_empty_list(states)
    no_country = is_empty_list(countries)

    if no_city and no_state and no_country:
        return 'none'

    if any(c != 'USA' for c in countries):
        if no_city and no_state:
            return 'international'
        return 'international'

    if no_city and not no_state and states[0] == pub_state:
        return 'state'

    if not no_city and not no_state and states[0] == pub_state:
        return 'local'

    return 'national'

df = pd.read_csv("../results/cc/cc-data.csv")

df['geo_cities'] = df['geo_cities'].apply(parse_list)
df['geo_states'] = df['geo_states'].apply(parse_list)
df['geo_countries'] = df['geo_countries'].apply(parse_list)

df['gfl_cc'] = df.apply(classify, axis=1)
df.to_csv("cc-gfl.csv", index=False)

df = pd.read_csv("../results/cc/cc-gfl.csv")

y_true = df['label']
y_pred = df['gfl_cc']

ordered_display = sorted(df['label'].unique())
ordered_numeric = ordered_display

labels = ["international", "national", "state", "local", "none"]
display_labels = ["intl.", "national", "state", "local", "none"]

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, labels=labels))

cm = confusion_matrix(y_true, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=display_labels,
    yticklabels=display_labels,
    cbar=True,
    linewidths=0.5,
    linecolor="gray",
    annot_kws={"fontsize": 48},
)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=28)

ax.set_xlabel("Predicted Label", fontsize=28)
ax.set_ylabel("True Label", fontsize=28)

plt.xticks(fontsize=28)
plt.yticks(rotation=90, fontsize=28)

plt.tight_layout()
plt.savefig("../results/cc/con_matrix_cc.png")
plt.show()
