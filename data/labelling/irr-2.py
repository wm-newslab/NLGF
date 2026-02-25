import pandas as pd
import ast
import numpy as np
from sklearn.metrics import cohen_kappa_score, jaccard_score
import krippendorff
import json

# ------------------------------------------------------------
# Step 1 — Load geo label universe using your function
# ------------------------------------------------------------

county_geojson = '../resources/county.geojson'
state_geojson = '../resources/state-us.geojson'
country_geojson = '../resources/countries.geojson'

def load_geoids(geojson_file, county_geojson, state_geojson):
    with open(geojson_file, encoding="utf-8") as f:
        geojson_data = json.load(f)

    geoids = set()
    for feature in geojson_data['features']:

        if geojson_file == county_geojson:
            geoid = feature['properties']['GEOID']

        elif geojson_file == state_geojson:
            geoid = feature['properties']['NAME']

        else:  # country level
            geoid = feature.get('id', feature['properties'].get('ISO_A3'))

        geoids.add(geoid)

    return geoids


def get_all_geo_labels(county_geojson, state_geojson, country_geojson):
    county_ids = load_geoids(county_geojson, county_geojson, state_geojson)
    state_ids = load_geoids(state_geojson, county_geojson, state_geojson)
    country_ids = load_geoids(country_geojson, county_geojson, state_geojson)

    all_geo_labels = county_ids.union(state_ids).union(country_ids)

    print("Total labels in universe:", len(all_geo_labels))
    print("Sample labels:", list(all_geo_labels)[:10])

    return sorted(all_geo_labels)


# Load full label universe
ALL_LABELS = get_all_geo_labels(county_geojson, state_geojson, country_geojson)
ALL_LABELS.append('none')

# ------------------------------------------------------------
# Step 2 — Load annotation data
# ------------------------------------------------------------

df = pd.read_csv("data_labels.csv")

# Convert stringified lists to Python lists
df["kate-gf"] = df["kate-gf"].apply(ast.literal_eval)
df["gaga-gf"] = df["gaga-gf"].apply(ast.literal_eval)

print("\nSample rows:\n", df.head())


# ------------------------------------------------------------
# Step 3 — Encode multi-label lists as binary multi-hot vectors
# ------------------------------------------------------------

label_index = {lab: i for i, lab in enumerate(ALL_LABELS)}

def encode_label_list(lst):
    vec = np.zeros(len(ALL_LABELS), dtype=int)
    for lab in lst:
        if lab in label_index:
            vec[label_index[lab]] = 1
        else:
            print(f"WARNING: Unknown label encountered: {lab}")
    return vec

kate_matrix = np.vstack(df["kate-gf"].apply(encode_label_list).to_numpy())
gaga_matrix = np.vstack(df["gaga-gf"].apply(encode_label_list).to_numpy())


# ------------------------------------------------------------
# Step 4 — Compute Per-Label Cohen’s Kappa
# ------------------------------------------------------------

print("\nComputing Cohen’s Kappa per label...")

kappa_per_label = {}

for i, lab in enumerate(ALL_LABELS):

    a = kate_matrix[:, i]
    b = gaga_matrix[:, i]

    # If either annotator has no variance → kappa undefined
    if (len(np.unique(a)) < 2) or (len(np.unique(b)) < 2):
        kappa = np.nan
    else:
        kappa = cohen_kappa_score(a, b, labels=[0, 1])

    kappa_per_label[lab] = kappa

kappa_df = (
    pd.DataFrame.from_dict(kappa_per_label, orient="index", columns=["kappa"])
      .dropna()
      .sort_values("kappa", ascending=False)
)

print("\nPer-label Cohen’s Kappa (non-NaN only):\n", kappa_df.head())

# ------------------------------------------------------------
# Step 5 — Macro-average Kappa
# ------------------------------------------------------------

macro_kappa = kappa_df["kappa"].mean()
print("\nMacro-average Kappa:", macro_kappa)


# ------------------------------------------------------------
# Step 6 — Per-item Jaccard similarity (set overlap measure)
# ------------------------------------------------------------

jaccards = [
    jaccard_score(kate_matrix[i], gaga_matrix[i], average="binary")
    for i in range(len(df))
]

mean_jaccard = np.mean(jaccards)
print("\nMean Jaccard Agreement:", mean_jaccard)


# ------------------------------------------------------------
# Step 7 — FIXED Krippendorff’s Alpha (nominal)
# ------------------------------------------------------------

print("\nComputing Krippendorff's Alpha (nominal)...")

# Flatten each annotator's full label vector
kate_flat = kate_matrix.flatten()
gaga_flat = gaga_matrix.flatten()

# Shape must be (num_raters, num_observations)
alpha_matrix = np.vstack([kate_flat, gaga_flat])

alpha = krippendorff.alpha(
    reliability_data=alpha_matrix,
    level_of_measurement="nominal"
)

print("\nKrippendorff’s Alpha:", alpha)
